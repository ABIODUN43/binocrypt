/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#0b0e14",
          card: "#121722",
          cardBorder: "#1e2638",
          cardHover: "#182030",
          accent: "#3b82f6",
          green: "#10b981",
          red: "#ef4444",
          amber: "#f59e0b",
          cyan: "#06b6d4",
          textMuted: "#94a3b8",
          textLight: "#f8fafc"
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
