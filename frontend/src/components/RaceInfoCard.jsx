import React from 'react'

export default function RaceInfoCard({
    raceName = 'JAPANESE GP',
    dates = 'OCTOBER 13-15',
    time = '06:00 UTC',
    track = 'Suzuka',
    country,
    circuitImage
}) {
    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '16px',
            overflow: 'hidden'
        }}>
            <div style={{
                fontSize: '11px',
                fontWeight: '700',
                letterSpacing: '1px',
                color: '#888888',
                textTransform: 'uppercase',
                marginBottom: '12px'
            }}>
                NEXT RACE
            </div>

            {/* Track Visualization */}
            <div style={{
                backgroundColor: '#0f0f0f',
                border: circuitImage ? 'none' : '1px solid #3a3a3a',
                borderRadius: '8px',
                height: '160px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '14px',
                color: '#888888',
                fontSize: '48px',
                position: 'relative',
                overflow: 'hidden',
                backgroundImage: circuitImage ? `url(${circuitImage})` : 'none',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat'
            }}>
                {!circuitImage && (
                    <div style={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px'
                    }}>
                        <div style={{
                            fontSize: '14px',
                            color: '#64748b',
                            fontWeight: '600',
                            letterSpacing: '1px'
                        }}>
                            {track === 'Loading...' ? 'LOADING CIRCUIT...' : track.toUpperCase()}
                        </div>
                        <div style={{
                            fontSize: '10px',
                            color: '#64748b',
                            opacity: 0.7
                        }}>
                            Circuit image loading
                        </div>
                    </div>
                )}
                {circuitImage && (
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'linear-gradient(to top, rgba(26, 26, 26, 0.9) 0%, rgba(26, 26, 26, 0.3) 50%, transparent 100%)',
                        display: 'flex',
                        alignItems: 'flex-end',
                        justifyContent: 'center',
                        padding: '12px'
                    }}>
                        <div style={{
                            fontSize: '14px',
                            fontWeight: '700',
                            color: '#00d4ff',
                            textTransform: 'uppercase',
                            letterSpacing: '1px',
                            textShadow: '0 2px 8px rgba(0, 0, 0, 0.8)'
                        }}>
                            {track}
                        </div>
                    </div>
                )}
            </div>

            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
            }}>
                <div>
                    <div style={{
                        color: '#888888',
                        fontSize: '9px',
                        fontWeight: '600',
                        letterSpacing: '0.5px',
                        marginBottom: '2px',
                        textTransform: 'uppercase'
                    }}>
                        Race
                    </div>
                    <div style={{
                        color: '#e2e8f0',
                        fontSize: '12px',
                        fontWeight: '700'
                    }}>
                        {raceName}
                    </div>
                    {country && (
                        <div style={{
                            color: '#888888',
                            fontSize: '10px',
                            marginTop: '2px'
                        }}>
                            📍 {country}
                        </div>
                    )}
                </div>

                <div>
                    <div style={{
                        color: '#888888',
                        fontSize: '9px',
                        fontWeight: '600',
                        letterSpacing: '0.5px',
                        marginBottom: '2px',
                        textTransform: 'uppercase'
                    }}>
                        Date
                    </div>
                    <div style={{
                        color: '#cbd5e1',
                        fontSize: '11px'
                    }}>
                        {dates}
                    </div>
                </div>

                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    color: '#cbd5e1',
                    fontSize: '11px'
                }}>
                    <span>🕐</span>
                    <span>{time}</span>
                </div>
            </div>

            <button style={{
                width: '100%',
                marginTop: '12px',
                backgroundColor: '#00d4ff',
                color: '#000',
                border: 'none',
                padding: '8px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: '700',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
            }}
                onMouseEnter={(e) => {
                    e.target.style.backgroundColor = '#00e5ff'
                    e.target.style.transform = 'translateY(-2px)'
                }}
                onMouseLeave={(e) => {
                    e.target.style.backgroundColor = '#00d4ff'
                    e.target.style.transform = 'translateY(0)'
                }}
            >
                VIEW FULL PREDICTIONS
            </button>
        </div>
    )
}
