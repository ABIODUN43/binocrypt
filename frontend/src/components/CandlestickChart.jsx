import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import { RefreshCw, Maximize2, ZoomIn, ZoomOut } from 'lucide-react';

export default function CandlestickChart({ symbol, setup }) {
  const [interval, setInterval] = useState('1h');
  const [candles, setCandles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hoveredCandle, setHoveredCandle] = useState(null);
  const svgRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    async function loadCandles() {
      if (!symbol) return;
      setIsLoading(true);
      try {
        const data = await api.getKlines(symbol, interval, 80);
        if (isMounted) {
          setCandles(data);
          setIsLoading(false);
        }
      } catch (err) {
        console.error("Failed to load candles:", err);
        if (isMounted) setIsLoading(false);
      }
    }
    loadCandles();
    return () => { isMounted = false; };
  }, [symbol, interval]);

  if (isLoading && candles.length === 0) {
    return (
      <div className="terminal-card h-80 flex items-center justify-center">
        <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
        <span className="text-xs text-gray-400 ml-2">Loading historical candles...</span>
      </div>
    );
  }

  // Chart Dimensions
  const width = 800;
  const height = 360;
  const padding = { top: 20, right: 60, bottom: 40, left: 10 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const validCandles = candles.slice(-60); // display last 60 candles
  if (validCandles.length === 0) {
    return (
      <div className="terminal-card h-80 flex items-center justify-center text-xs text-gray-500">
        No candle data available for {symbol}
      </div>
    );
  }

  const highs = validCandles.map(c => c.high);
  const lows = validCandles.map(c => c.low);
  const minPrice = Math.min(...lows) * 0.998;
  const maxPrice = Math.max(...highs) * 1.002;
  const priceRange = maxPrice - minPrice || 1;

  const volumes = validCandles.map(c => c.volume);
  const maxVolume = Math.max(...volumes) || 1;

  const getX = (index) => padding.left + (index * (plotWidth / validCandles.length)) + ((plotWidth / validCandles.length) / 2);
  const getY = (price) => padding.top + plotHeight - ((price - minPrice) / priceRange) * plotHeight;
  const candleWidth = Math.max(3, (plotWidth / validCandles.length) * 0.65);

  const formatPrice = (p) => p < 1 ? p.toFixed(4) : p.toFixed(2);

  return (
    <div className="terminal-card p-4">
      {/* Chart Top Header Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e2638] pb-3 mb-2">
        <div className="flex items-center gap-3">
          <span className="font-bold text-white text-sm">{symbol}</span>
          {hoveredCandle ? (
            <div className="flex items-center gap-2 text-[11px] font-mono-nums text-gray-400">
              <span>O: <strong className="text-white">${formatPrice(hoveredCandle.open)}</strong></span>
              <span>H: <strong className="text-emerald-400">${formatPrice(hoveredCandle.high)}</strong></span>
              <span>L: <strong className="text-rose-400">${formatPrice(hoveredCandle.low)}</strong></span>
              <span>C: <strong className={hoveredCandle.close >= hoveredCandle.open ? "text-emerald-400" : "text-rose-400"}>${formatPrice(hoveredCandle.close)}</strong></span>
              <span className="hidden sm:inline">Vol: {hoveredCandle.volume.toFixed(1)}</span>
            </div>
          ) : (
            <span className="text-xs text-gray-500">Hover candles for details</span>
          )}
        </div>

        {/* Timeframe selector */}
        <div className="flex items-center bg-[#0b0e14] border border-[#1e2638] rounded p-0.5 text-xs font-mono">
          {['15m', '1h', '4h', '1d'].map((tf) => (
            <button
              key={tf}
              onClick={() => setInterval(tf)}
              className={`px-2.5 py-1 rounded transition ${
                interval === tf 
                  ? 'bg-blue-600 text-white font-bold' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tf.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative w-full overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-72 sm:h-80 select-none"
          onMouseLeave={() => setHoveredCandle(null)}
        >
          {/* Background Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const y = padding.top + (plotHeight * pct);
            const p = maxPrice - (priceRange * pct);
            return (
              <g key={idx}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="#1e2638"
                  strokeDasharray="3,3"
                />
                <text
                  x={width - padding.right + 6}
                  y={y + 3}
                  fill="#64748b"
                  fontSize="10"
                  fontFamily="JetBrains Mono"
                >
                  ${formatPrice(p)}
                </text>
              </g>
            );
          })}

          {/* Volume bars (bottom 20% of plot) */}
          {validCandles.map((c, idx) => {
            const x = getX(idx);
            const volHeight = (c.volume / maxVolume) * (plotHeight * 0.22);
            const y = padding.top + plotHeight - volHeight;
            const isBull = c.close >= c.open;
            return (
              <rect
                key={`vol-${idx}`}
                x={x - candleWidth / 2}
                y={y}
                width={candleWidth}
                height={volHeight}
                fill={isBull ? "#10b981" : "#ef4444"}
                opacity="0.25"
              />
            );
          })}

          {/* Candlesticks (wicks & bodies) */}
          {validCandles.map((c, idx) => {
            const x = getX(idx);
            const isBull = c.close >= c.open;
            const openY = getY(c.open);
            const closeY = getY(c.close);
            const highY = getY(c.high);
            const lowY = getY(c.low);

            const bodyTop = Math.min(openY, closeY);
            const bodyHeight = Math.max(1.5, Math.abs(closeY - openY));
            const color = isBull ? "#10b981" : "#ef4444";

            return (
              <g 
                key={`c-${idx}`}
                className="cursor-crosshair"
                onMouseEnter={() => setHoveredCandle(c)}
              >
                {/* Wick */}
                <line
                  x1={x}
                  y1={highY}
                  x2={x}
                  y2={lowY}
                  stroke={color}
                  strokeWidth="1.2"
                />
                {/* Body */}
                <rect
                  x={x - candleWidth / 2}
                  y={bodyTop}
                  width={candleWidth}
                  height={bodyHeight}
                  fill={color}
                  rx="0.5"
                />
              </g>
            );
          })}

          {/* Trade Setup Overlays (Entry, Stop, TP1, TP2) */}
          {setup && setup.symbol === symbol && (
            <g>
              {/* Stop Loss Line */}
              {setup.stop_loss >= minPrice && setup.stop_loss <= maxPrice && (
                <g>
                  <line
                    x1={padding.left}
                    y1={getY(setup.stop_loss)}
                    x2={width - padding.right}
                    y2={getY(setup.stop_loss)}
                    stroke="#ef4444"
                    strokeWidth="1.5"
                    strokeDasharray="4,4"
                  />
                  <text
                    x={width - padding.right + 6}
                    y={getY(setup.stop_loss) + 3}
                    fill="#ef4444"
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="JetBrains Mono"
                  >
                    SL ${formatPrice(setup.stop_loss)}
                  </text>
                </g>
              )}

              {/* Entry Zone Line */}
              {setup.current_price >= minPrice && setup.current_price <= maxPrice && (
                <g>
                  <line
                    x1={padding.left}
                    y1={getY(setup.current_price)}
                    x2={width - padding.right}
                    y2={getY(setup.current_price)}
                    stroke="#3b82f6"
                    strokeWidth="1.5"
                    strokeDasharray="4,4"
                  />
                  <text
                    x={width - padding.right + 6}
                    y={getY(setup.current_price) + 3}
                    fill="#3b82f6"
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="JetBrains Mono"
                  >
                    ENTRY ${formatPrice(setup.current_price)}
                  </text>
                </g>
              )}

              {/* Target 1 Line */}
              {setup.tp1 >= minPrice && setup.tp1 <= maxPrice && (
                <g>
                  <line
                    x1={padding.left}
                    y1={getY(setup.tp1)}
                    x2={width - padding.right}
                    y2={getY(setup.tp1)}
                    stroke="#10b981"
                    strokeWidth="1.5"
                    strokeDasharray="4,4"
                  />
                  <text
                    x={width - padding.right + 6}
                    y={getY(setup.tp1) + 3}
                    fill="#10b981"
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="JetBrains Mono"
                  >
                    TP1 ${formatPrice(setup.tp1)}
                  </text>
                </g>
              )}

              {/* Target 2 Line */}
              {setup.tp2 >= minPrice && setup.tp2 <= maxPrice && (
                <g>
                  <line
                    x1={padding.left}
                    y1={getY(setup.tp2)}
                    x2={width - padding.right}
                    y2={getY(setup.tp2)}
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                    strokeDasharray="4,4"
                  />
                  <text
                    x={width - padding.right + 6}
                    y={getY(setup.tp2) + 3}
                    fill="#06b6d4"
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="JetBrains Mono"
                  >
                    TP2 ${formatPrice(setup.tp2)}
                  </text>
                </g>
              )}
            </g>
          )}

        </svg>
      </div>
    </div>
  );
}
