import React, { useState } from 'react'

export default function MatchupCard({ drivers = [] }) {
    const [driver1, setDriver1] = useState(drivers[0] || null)
    const [driver2, setDriver2] = useState(drivers[1] || null)

    const mockDrivers = [
        { id: 1, name: 'Max Verstappen', team: 'Red Bull', color: '#1e3a8a', odds: 85 },
        { id: 2, name: 'Lando Norris', team: 'McLaren', color: '#ea580c', odds: 72 },
        { id: 3, name: 'Charles Leclerc', team: 'Ferrari', color: '#dc2626', odds: 65 },
        { id: 4, name: 'Lewis Hamilton', team: 'Mercedes', color: '#64748b', odds: 58 },
        { id: 5, name: 'Carlos Sainz', team: 'Ferrari', color: '#dc2626', odds: 52 }
    ]

    const selectedDriver1 = driver1 || mockDrivers[0]
    const selectedDriver2 = driver2 || mockDrivers[1]

    const total = selectedDriver1.odds + selectedDriver2.odds
    const p1Percentage = (selectedDriver1.odds / total) * 100
    const p2Percentage = (selectedDriver2.odds / total) * 100

    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '18px'
        }}>
            <div style={{
                fontSize: '11px',
                fontWeight: '700',
                letterSpacing: '1px',
                color: '#888888',
                textTransform: 'uppercase',
                marginBottom: '16px'
            }}>
                HEAD-TO-HEAD MATCHUP
            </div>

            {/* Driver Selectors */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr auto 1fr',
                gap: '12px',
                marginBottom: '16px',
                alignItems: 'end'
            }}>
                <div>
                    <label style={{
                        fontSize: '9px',
                        fontWeight: '600',
                        color: '#888888',
                        display: 'block',
                        marginBottom: '6px'
                    }}>
                        SELECT DRIVER 1
                    </label>
                    <select
                        value={selectedDriver1.id}
                        onChange={(e) => setDriver1(mockDrivers.find(d => d.id === parseInt(e.target.value)))}
                        style={{
                            width: '100%',
                            backgroundColor: '#0f0f0f',
                            color: '#e2e8f0',
                            border: '1px solid #3a3a3a',
                            padding: '8px',
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '600',
                            cursor: 'pointer'
                        }}
                    >
                        {mockDrivers.map(driver => (
                            <option key={driver.id} value={driver.id} style={{ backgroundColor: '#1a1a1a' }}>
                                {driver.name}
                            </option>
                        ))}
                    </select>
                </div>

                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '14px',
                    fontWeight: '800',
                    color: '#888888',
                    padding: '8px 12px',
                    backgroundColor: 'rgba(80, 80, 80, 0.3)',
                    borderRadius: '6px'
                }}>
                    VS
                </div>

                <div>
                    <label style={{
                        fontSize: '9px',
                        fontWeight: '600',
                        color: '#888888',
                        display: 'block',
                        marginBottom: '6px'
                    }}>
                        SELECT DRIVER 2
                    </label>
                    <select
                        value={selectedDriver2.id}
                        onChange={(e) => setDriver2(mockDrivers.find(d => d.id === parseInt(e.target.value)))}
                        style={{
                            width: '100%',
                            backgroundColor: '#0f0f0f',
                            color: '#e2e8f0',
                            border: '1px solid #3a3a3a',
                            padding: '8px',
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '600',
                            cursor: 'pointer'
                        }}
                    >
                        {mockDrivers.map(driver => (
                            <option key={driver.id} value={driver.id} style={{ backgroundColor: '#1a1a1a' }}>
                                {driver.name}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Comparison Display */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                marginBottom: '16px'
            }}>
                {/* Driver 1 */}
                <div style={{
                    backgroundColor: '#0f0f0f',
                    border: `2px solid ${selectedDriver1.color}`,
                    borderRadius: '8px',
                    padding: '12px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '50px',
                        height: '50px',
                        margin: '0 auto 8px',
                        borderRadius: '50%',
                        backgroundColor: selectedDriver1.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '24px'
                    }}>
                        👤
                    </div>
                    <div style={{
                        fontSize: '11px',
                        fontWeight: '700',
                        color: '#e2e8f0',
                        marginBottom: '4px'
                    }}>
                        {selectedDriver1.name.split(' ')[1]}
                    </div>
                    <div style={{
                        fontSize: '9px',
                        color: '#888888',
                        marginBottom: '8px'
                    }}>
                        {selectedDriver1.team}
                    </div>
                    <div style={{
                        fontSize: '20px',
                        fontWeight: '800',
                        color: selectedDriver1.color
                    }}>
                        {p1Percentage.toFixed(0)}%
                    </div>
                </div>

                {/* Driver 2 */}
                <div style={{
                    backgroundColor: '#0f0f0f',
                    border: `2px solid ${selectedDriver2.color}`,
                    borderRadius: '8px',
                    padding: '12px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '50px',
                        height: '50px',
                        margin: '0 auto 8px',
                        borderRadius: '50%',
                        backgroundColor: selectedDriver2.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '24px'
                    }}>
                        👤
                    </div>
                    <div style={{
                        fontSize: '11px',
                        fontWeight: '700',
                        color: '#e2e8f0',
                        marginBottom: '4px'
                    }}>
                        {selectedDriver2.name.split(' ')[1]}
                    </div>
                    <div style={{
                        fontSize: '9px',
                        color: '#888888',
                        marginBottom: '8px'
                    }}>
                        {selectedDriver2.team}
                    </div>
                    <div style={{
                        fontSize: '20px',
                        fontWeight: '800',
                        color: selectedDriver2.color
                    }}>
                        {p2Percentage.toFixed(0)}%
                    </div>
                </div>
            </div>

            {/* Comparison Bars */}
            <div style={{
                marginBottom: '12px'
            }}>
                <div style={{
                    display: 'flex',
                    height: '12px',
                    backgroundColor: 'rgba(80, 80, 80, 0.5)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                    gap: '2px'
                }}>
                    <div style={{
                        flex: p1Percentage,
                        backgroundColor: selectedDriver1.color,
                        borderRadius: '6px',
                        transition: 'flex 0.3s ease'
                    }} />
                    <div style={{
                        flex: p2Percentage,
                        backgroundColor: selectedDriver2.color,
                        borderRadius: '6px',
                        transition: 'flex 0.3s ease'
                    }} />
                </div>
            </div>

            {/* Prediction Text */}
            <div style={{
                padding: '12px',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderRadius: '6px',
                borderLeft: '3px solid #00d4ff',
                fontSize: '10px',
                color: '#cbd5e1',
                lineHeight: '1.5'
            }}>
                Based on current form, <strong>{selectedDriver1.name}</strong> has a <strong>{p1Percentage.toFixed(0)}%</strong> chance of outperforming <strong>{selectedDriver2.name}</strong> in this race.
            </div>
        </div>
    )
}
