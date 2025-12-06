/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'serenity-cyan': '#3BC9DB',
        'serenity-ocean': '#1971C2',
        'serenity-mist': '#E7F5FF',
        'serenity-green': '#51CF66',
        'serenity-lavender': '#B197FC',
        'serenity-sand': '#FFE8CC',
        'serenity-midnight': '#0B1728',
        'serenity-slate': '#475569',
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        body: ['"DM Sans"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glow': '0 0 40px rgba(125, 211, 252, 0.5)',
        'healing': '0 0 30px rgba(34, 211, 238, 0.4)',
        'sm': '0 2px 8px rgba(15, 23, 42, 0.08)',
        'md': '0 4px 16px rgba(15, 23, 42, 0.12)',
        'lg': '0 12px 32px rgba(15, 23, 42, 0.15)',
      },
      backdropBlur: {
        'xl': '20px',
      },
      borderRadius: {
        '3xl': '24px',
        '4xl': '32px',
      },
      animation: {
        'breathe': 'breathe 4s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'shimmer': 'shimmer 3s infinite',
        'gradient-shift': 'gradient-shift 8s ease infinite',
      },
      transitionTimingFunction: {
        'expo-out': 'cubic-bezier(0.19, 1, 0.22, 1)',
        'quart': 'cubic-bezier(0.76, 0, 0.24, 1)',
      },
    },
  },
  plugins: [],
}