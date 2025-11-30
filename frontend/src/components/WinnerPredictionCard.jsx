import React from 'react'

export default function WinnerPredictionCard({
    percentage = 72,
    driverName = 'Max Verstappen',
    teamColor = '#1e3a8a',
    headshotUrl = null,
    fullName = null,
    confidence = 'HIGH',
    confidenceColor = '#f59e0b'
}) {
    const displayName = fullName || driverName;

    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '18px',
            borderLeft: `4px solid ${confidenceColor}`,
            position: 'relative',
            overflow: 'hidden',
            backgroundImage: headshotUrl ? `url(${headshotUrl})` : `linear-gradient(135deg, ${teamColor}33 0%, ${teamColor}11 100%)`,
            backgroundSize: headshotUrl ? 'auto 100%' : 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
            minHeight: '250px',
            display: 'flex',
            flexDirection: 'column'
        }}>
            {/* Dark overlay for readability */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: headshotUrl ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.4)',
                zIndex: 0
            }} />

            {/* Background accent */}
            <div style={{
                position: 'absolute',
                top: 0,
                right: 0,
                width: '120px',
                height: '120px',
                backgroundColor: `rgba(0, 212, 255, 0.05)`,
                borderRadius: '50%',
                transform: 'translate(40px, -40px)',
                pointerEvents: 'none',
                zIndex: 1
            }} />

            <div style={{
                fontSize: '11px',
                fontWeight: '700',
                letterSpacing: '1px',
                color: '#888888',
                textTransform: 'uppercase',
                marginBottom: '14px',
                position: 'relative',
                zIndex: 2
            }}>
                RACE WINNER PREDICTION
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-end',
                marginTop: 'auto',
                position: 'relative',
                zIndex: 2
            }}>
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start'
                }}>
                    <div style={{
                        fontSize: '56px',
                        fontWeight: '800',
                        color: '#ffffff',
                        lineHeight: '1',
                        marginBottom: '6px'
                    }}>
                        {percentage}%
                    </div>
                    <div style={{
                        fontSize: '9px',
                        fontWeight: '700',
                        color: confidence === 'HIGH' ? confidenceColor : '#ffffff',
                        letterSpacing: '1px',
                        marginBottom: '10px'
                    }}>
                        {confidence} CONFIDENCE
                    </div>
                    <div style={{
                        height: '3px',
                        width: '60px',
                        background: `linear-gradient(90deg, ${confidenceColor} 0%, transparent 100%)`,
                        borderRadius: '1.5px'
                    }} />
                </div>

                {/* Driver Name - Bottom Right */}
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-end'
                }}>
                    <div style={{
                        fontSize: displayName.includes(' ') ? '56px' : '42px',
                        fontWeight: '800',
                        color: '#ffffff',
                        lineHeight: '1',
                        marginBottom: '0px'
                    }}>
                        {displayName.includes(' ') ? displayName.split(' ')[0] : displayName}
                    </div>
                    {displayName.includes(' ') && (
                        <div style={{
                            fontSize: '9px',
                            fontWeight: '700',
                            color: '#ffffff',
                            letterSpacing: '1px'
                        }}>
                            {displayName.split(' ')[1]}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
