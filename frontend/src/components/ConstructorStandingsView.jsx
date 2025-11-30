import React, { useState, useEffect } from 'react'
import { fetchConstructorStandings } from '../api'

export default function ConstructorStandingsView() {
    const [constructorStandings, setConstructorStandings] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadStandings() {
            try {
                console.log('Loading constructor standings...')
                const standings = await fetchConstructorStandings()
                console.log('✓ Constructor standings loaded:', standings)
                console.log('  Total constructors:', standings.length)
                if (standings.length > 0) {
                    console.log('  First constructor:', standings[0])
                    console.log('  First constructor points:', standings[0].points)
                    console.log('  First constructor predicted:', standings[0].predictedPoints)
                }
                setConstructorStandings(standings)
            } catch (error) {
                console.error('❌ Failed to load standings:', error)
                // Fallback data
                setConstructorStandings([
                    { position: 1, constructorName: 'McLaren', points: 756, predictedPoints: 785, wins: 14, teamColor: '#ea580c' },
                    { position: 2, constructorName: 'Mercedes', points: 398, predictedPoints: 463, wins: 2, teamColor: '#64748b' },
                    { position: 3, constructorName: 'Red Bull', points: 366, predictedPoints: 429, wins: 5, teamColor: '#0052cc' },
                    { position: 4, constructorName: 'Ferrari', points: 342, predictedPoints: 378, wins: 0, teamColor: '#dc2626' },
                    { position: 5, constructorName: 'Aston Martin', points: 95, predictedPoints: 110, wins: 0, teamColor: '#006241' },
                ])
            } finally {
                setLoading(false)
            }
        }

        loadStandings()
    }, [])

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
                CONSTRUCTOR STANDINGS
            </div>

            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1px',
                backgroundColor: '#0f0f0f',
                borderRadius: '8px',
                overflow: 'hidden'
            }}>
                {/* Header Row */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '40px 1fr 80px 100px 60px',
                    gap: '12px',
                    padding: '12px',
                    backgroundColor: 'rgba(80, 80, 80, 0.3)',
                    borderBottom: '1px solid #3a3a3a',
                    fontSize: '10px',
                    fontWeight: '700',
                    color: '#888888',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                }}>
                    <div>POS</div>
                    <div>TEAM</div>
                    <div>ACTUAL PTS</div>
                    <div>PREDICTED</div>
                    <div>WINS</div>
                </div>

                {loading ? (
                    <div style={{
                        padding: '40px',
                        textAlign: 'center',
                        color: '#888888',
                        fontSize: '12px'
                    }}>
                        Loading standings...
                    </div>
                ) : (
                    <>
                        {/* Data Rows */}
                        {constructorStandings.map((constructor, idx) => (
                            <div
                                key={constructor.position}
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: '40px 1fr 80px 100px 60px',
                                    gap: '12px',
                                    padding: '12px',
                                    backgroundColor: idx % 2 === 0 ? 'rgba(80, 80, 80, 0.15)' : 'transparent',
                                    borderBottom: idx < constructorStandings.length - 1 ? '1px solid #3a3a3a' : 'none',
                                    alignItems: 'center'
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = 'rgba(0, 212, 255, 0.2)'
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = idx % 2 === 0 ? 'rgba(80, 80, 80, 0.15)' : 'transparent'
                                }}
                            >
                                <div style={{
                                    fontSize: '13px',
                                    fontWeight: '800',
                                    color: constructor.position === 1 ? '#fbbf24' : '#cbd5e1'
                                }}>
                                    {constructor.position}
                                </div>
                                <div>
                                    <div style={{
                                        fontSize: '11px',
                                        fontWeight: '700',
                                        color: '#e2e8f0',
                                        marginBottom: '2px'
                                    }}>
                                        {constructor.constructorName}
                                    </div>
                                </div>
                                <div style={{
                                    fontSize: '12px',
                                    fontWeight: '700',
                                    color: '#00d4ff'
                                }}>
                                    {constructor.points}
                                </div>
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px'
                                }}>
                                    <div style={{
                                        fontSize: '12px',
                                        fontWeight: '700',
                                        color: '#ea580c'
                                    }}>
                                        {constructor.predictedPoints}
                                    </div>
                                    <div style={{
                                        fontSize: '9px',
                                        color: constructor.predictedPoints > constructor.points ? '#10b981' : '#888888',
                                        fontWeight: '600'
                                    }}>
                                        {constructor.predictedPoints > constructor.points ? '↑' : '→'}
                                    </div>
                                </div>
                                <div style={{
                                    fontSize: '12px',
                                    fontWeight: '700',
                                    color: '#888888'
                                }}>
                                    {constructor.wins}
                                </div>
                            </div>
                        ))}
                    </>
                )}
            </div>

            <div style={{
                marginTop: '16px',
                padding: '12px',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderRadius: '6px',
                fontSize: '10px',
                color: '#cbd5e1',
                lineHeight: '1.6'
            }}>
                <strong style={{ color: '#00d4ff' }}>Model Accuracy:</strong> Predictions based on driver performance aggregation. 22 of 24 races completed.
            </div>
        </div>
    )
}
