import React, { useState } from 'react'

export default function DriverCard({
    name,
    team,
    percentage,
    teamColor,
    position,
    points,
    onHover,
    headshotUrl,
    fullName,
    confidence = 'MEDIUM',
    confidenceColor = '#f59e0b'
}) {
    const [isHovered, setIsHovered] = useState(false)

    let barColor = confidenceColor

    return (
        <div
            onMouseEnter={() => {
                setIsHovered(true)
                onHover?.()
            }}
            onMouseLeave={() => setIsHovered(false)}
            style={{
                backgroundColor: isHovered ? '#2a2a2a' : '#1a1a1a',
                border: isHovered ? '1px solid #00d4ff' : '1px solid #3a3a3a',
                borderRadius: '10px',
                padding: '14px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                transform: isHovered ? 'translateY(-4px)' : 'translateY(0)',
                boxShadow: isHovered ? '0 8px 16px rgba(0, 212, 255, 0.2)' : 'none'
            }}>
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
            }}>
                <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    backgroundColor: teamColor || '#334155',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#fff',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    flexShrink: 0,
                    overflow: 'hidden',
                    border: '2px solid rgba(255, 255, 255, 0.1)'
                }}>
                    {headshotUrl ? (
                        <img
                            src={headshotUrl}
                            alt={fullName || name}
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                            }}
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.parentElement.textContent = name.charAt(0);
                            }}
                        />
                    ) : (
                        name.charAt(0)
                    )}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{
                        color: '#e2e8f0',
                        fontSize: '13px',
                        fontWeight: '700',
                        letterSpacing: '0.5px'
                    }}>
                        {fullName || name}
                    </div>
                    <div style={{
                        color: '#888888',
                        fontSize: '10px',
                        marginTop: '2px'
                    }}>
                        {team}
                    </div>
                </div>
                {position && (
                    <div style={{
                        fontSize: '12px',
                        fontWeight: '800',
                        color: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        padding: '4px 8px',
                        borderRadius: '4px'
                    }}>
                        P{position}
                    </div>
                )}
            </div>

            <div style={{
                height: '6px',
                backgroundColor: 'rgba(80, 80, 80, 0.6)',
                borderRadius: '3px',
                overflow: 'hidden'
            }}>
                <div style={{
                    height: '100%',
                    width: `${percentage}%`,
                    backgroundColor: barColor,
                    borderRadius: '3px',
                    transition: 'width 0.3s ease'
                }} />
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div style={{
                    fontSize: '13px',
                    fontWeight: '700',
                    color: barColor
                }}>
                    {percentage}%
                </div>
                <div style={{
                    fontSize: '9px',
                    fontWeight: '600',
                    color: '#888888',
                    letterSpacing: '0.5px'
                }}>
                    {confidence}
                </div>
            </div>

            {isHovered && (
                <div style={{
                    marginTop: '8px',
                    paddingTop: '8px',
                    borderTop: '1px solid rgba(80, 80, 80, 0.5)',
                    fontSize: '10px',
                    color: '#cbd5e1'
                }}>
                    {points && <div>Points: <span style={{ color: '#00d4ff', fontWeight: '700' }}>{points}</span></div>}
                    <div style={{ marginTop: '4px' }}>
                        Click for detailed analysis →
                    </div>
                </div>
            )}
        </div>
    )
}
