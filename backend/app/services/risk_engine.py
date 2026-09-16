from typing import Dict, Any, Tuple
from ..core.config import settings
from ..models.schemas import RiskSizingResponse

class RiskEngine:
    @staticmethod
    def calculate_position_size(
        account_equity: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float
    ) -> RiskSizingResponse:
        """
        Calculates position size strictly from stop-loss distance:
        Position Size ($) = Risk Amount ($) / Stop Distance
        """
        if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
            return RiskSizingResponse(
                account_equity=account_equity,
                risk_pct=risk_pct,
                risk_amount_usd=0.0,
                stop_distance_pct=0.0,
                recommended_position_usd=0.0,
                recommended_quantity=0.0,
                is_valid=False,
                warning_or_notes="Invalid prices: Stop loss must be strictly below entry price."
            )

        # Risk amount in USD (e.g. $40 * 2% = $0.80)
        risk_amount_usd = account_equity * (risk_pct / 100.0)
        
        # Stop distance ratio
        stop_distance_ratio = (entry_price - stop_loss) / entry_price
        stop_distance_pct = round(stop_distance_ratio * 100.0, 2)

        # Calculated position size
        raw_position_usd = risk_amount_usd / stop_distance_ratio

        # Enforce account constraints
        max_notional_cap = account_equity * 0.95 # keep 5% buffer for fees/slippage
        
        is_valid = True
        notes = []

        if raw_position_usd > max_notional_cap:
            recommended_usd = max_notional_cap
            actual_risk = recommended_usd * stop_distance_ratio
            notes.append(f"Position capped at {max_notional_cap:.2f} USD (95% of equity). Actual risk: ${actual_risk:.2f} ({(actual_risk/account_equity)*100:.1f}%).")
        elif raw_position_usd < settings.MIN_ORDER_NOTIONAL:
            # Check if bumping to $5 min order stays within risk budget
            min_order_risk = settings.MIN_ORDER_NOTIONAL * stop_distance_ratio
            if min_order_risk <= (risk_amount_usd * 1.25):
                recommended_usd = settings.MIN_ORDER_NOTIONAL
                notes.append(f"Adjusted to exchange minimum order (${settings.MIN_ORDER_NOTIONAL:.2f} USDT). Expected risk at stop: ${min_order_risk:.2f}.")
            else:
                recommended_usd = settings.MIN_ORDER_NOTIONAL
                notes.append(f"Caution: ${settings.MIN_ORDER_NOTIONAL:.2f} minimum order exceeds standard 2% risk budget for this wide stop.")
        else:
            recommended_usd = round(raw_position_usd, 2)
            notes.append(f"Optimal risk-adjusted size: 2% risk (${risk_amount_usd:.2f}) on {stop_distance_pct}% stop.")

        recommended_qty = round(recommended_usd / entry_price, 6)

        return RiskSizingResponse(
            account_equity=round(account_equity, 2),
            risk_pct=risk_pct,
            risk_amount_usd=round(risk_amount_usd, 2),
            stop_distance_pct=stop_distance_pct,
            recommended_position_usd=round(recommended_usd, 2),
            recommended_quantity=recommended_qty,
            is_valid=is_valid,
            warning_or_notes=" | ".join(notes)
        )

risk_engine = RiskEngine()
