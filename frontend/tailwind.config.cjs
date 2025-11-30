/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './index.html',
        './src/**/*.{js,jsx,ts,tsx,html}'
    ],
    theme: {
        extend: {
            colors: {
                'f1-dark': '#0b0f14'
            }
        }
    },
    plugins: []
}
