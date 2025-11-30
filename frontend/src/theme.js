/**
 * F1 PREDICT - Design System & Theme Configuration
 * Centralized color palette, typography, and spacing constants
 */

// ===== COLOR PALETTE =====
export const colors = {
    // Background
    bg: {
        primary: '#1a1a1a',      // Main background (dark gray)
        secondary: '#2a2a2a',    // Card background
        tertiary: '#242424',     // Alternate background
        hover: '#333333',        // Hover state
    },

    // Text
    text: {
        primary: '#e2e8f0',      // Main text
        secondary: '#cbd5e1',    // Secondary text
        tertiary: '#94a3b8',     // Tertiary text
        muted: '#64748b',        // Muted text
    },

    // Borders
    border: {
        primary: '#3a3a3a',      // Main border
        light: 'rgba(100, 100, 100, 0.3)',
        hover: '#00d4ff',        // Hover border
    },

    // Accents
    accent: {
        cyan: '#00d4ff',         // Primary accent (brighter cyan)
        lightCyan: '#00e5ff',    // Light accent
        red: '#ef4444',          // Red
        orange: '#ea580c',       // Orange/McLaren
        amber: '#f59e0b',        // Amber
        pink: '#f43f5e',         // Pink
        green: '#10b981',        // Green
        yellow: '#fbbf24',       // Yellow
    },

    // Team Colors
    teams: {
        redBull: '#1e3a8a',      // Navy Blue
        mercedes: '#64748b',     // Silver Gray
        ferrari: '#dc2626',      // Ferrari Red
        mclaren: '#ea580c',      // McLaren Orange
        alpine: '#2563eb',       // Alpine Blue
        astonMartin: '#16a34a',  // Aston Martin Green
        haas: '#ffffff',         // Haas White
        alphaTauri: '#0f172a',   // AlphaTauri Navy
        williams: '#005aff',     // Williams Blue
        sauber: '#00d4ff',       // Sauber Cyan
    },

    // Semantic
    semantic: {
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#06b6d4',
    }
}

// ===== TYPOGRAPHY =====
export const typography = {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: {
        xs: '8px',
        sm: '9px',
        base: '10px',
        md: '11px',
        lg: '12px',
        xl: '13px',
        '2xl': '14px',
        '3xl': '16px',
        '4xl': '18px',
        '5xl': '20px',
        '6xl': '24px',
        '7xl': '28px',
        '8xl': '32px',
        display: '56px',
        hero: '60px',
    },
    fontWeight: {
        light: 300,
        normal: 400,
        semibold: 600,
        bold: 700,
        extrabold: 800,
    },
    letterSpacing: {
        tight: '-0.5px',
        normal: '0px',
        wide: '0.5px',
        wider: '1px',
        widest: '2px',
    }
}

// ===== SPACING =====
export const spacing = {
    xs: '4px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    '2xl': '24px',
    '3xl': '28px',
    '4xl': '32px',
}

// ===== BORDER RADIUS =====
export const borderRadius = {
    sm: '4px',
    md: '6px',
    lg: '8px',
    xl: '10px',
    full: '50%',
}

// ===== TRANSITIONS =====
export const transitions = {
    fast: '0.15s ease',
    normal: '0.3s ease',
    slow: '0.5s ease',
    verySlow: '1s ease',
}

// ===== SHADOWS =====
export const shadows = {
    sm: '0 2px 4px rgba(0, 0, 0, 0.1)',
    md: '0 4px 8px rgba(0, 0, 0, 0.15)',
    lg: '0 8px 16px rgba(6, 182, 212, 0.2)',
    xl: '0 12px 24px rgba(6, 182, 212, 0.3)',
}

// ===== BREAKPOINTS =====
export const breakpoints = {
    mobile: '480px',
    tablet: '768px',
    desktop: '1024px',
    wide: '1280px',
    ultrawide: '1600px',
}

// ===== COMMON STYLES =====
export const commonStyles = {
    card: {
        backgroundColor: colors.bg.secondary,
        border: `1px solid ${colors.border.primary}`,
        borderRadius: borderRadius.lg,
        padding: spacing.lg,
    },

    cardHover: {
        backgroundColor: colors.bg.hover,
        borderColor: colors.accent.cyan,
        boxShadow: shadows.lg,
        transition: `all ${transitions.normal}`,
    },

    label: {
        fontSize: typography.fontSize.sm,
        fontWeight: typography.fontWeight.bold,
        letterSpacing: typography.letterSpacing.wider,
        color: colors.text.tertiary,
        textTransform: 'uppercase',
    },

    badge: {
        padding: '4px 8px',
        borderRadius: borderRadius.md,
        fontSize: typography.fontSize.sm,
        fontWeight: typography.fontWeight.bold,
        display: 'inline-block',
    },

    progressBar: {
        height: '6px',
        backgroundColor: colors.border.primary,
        borderRadius: borderRadius.md,
        overflow: 'hidden',
    },
}

// ===== HELPER FUNCTIONS =====
export const getTeamColor = (teamName) => {
    const teamColorMap = {
        'Red Bull Racing': colors.teams.redBull,
        'Mercedes': colors.teams.mercedes,
        'Ferrari': colors.teams.ferrari,
        'McLaren': colors.teams.mclaren,
        'Alpine': colors.teams.alpine,
        'Aston Martin': colors.teams.astonMartin,
        'Haas': colors.teams.haas,
        'AlphaTauri': colors.teams.alphaTauri,
        'Williams': colors.teams.williams,
        'Sauber': colors.teams.sauber,
    }
    return teamColorMap[teamName] || colors.accent.cyan
}

export const getConfidenceColor = (percentage) => {
    if (percentage >= 75) return colors.accent.green
    if (percentage >= 60) return colors.accent.amber
    if (percentage >= 40) return colors.accent.pink
    return colors.accent.red
}

export const getConfidenceLevel = (percentage) => {
    if (percentage >= 80) return 'VERY HIGH'
    if (percentage >= 60) return 'HIGH'
    if (percentage >= 40) return 'MODERATE'
    return 'LOW'
}

// ===== ANIMATION KEYFRAMES =====
export const animations = `
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes slideIn {
    from { 
      opacity: 0; 
      transform: translateX(-20px); 
    }
    to { 
      opacity: 1; 
      transform: translateX(0); 
    }
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes scaleIn {
    from { 
      opacity: 0; 
      transform: scale(0.95); 
    }
    to { 
      opacity: 1; 
      transform: scale(1); 
    }
  }
`
