import React, { useState, useEffect } from 'react'

export default function RaceHistoryCard({ raceHistory = [] }) {
    const [races, setRaces] = useState([])
    const [sortBy, setSortBy] = useState('date')
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (raceHistory && raceHistory.length > 0) {
            // Sort races by date descending (most recent first)
            const sorted = [...raceHistory].sort((a, b) => {
                if (sortBy === 'date') {
                    return new Date(b.date) - new Date(a.date)
                } else if (sortBy === 'result') {
                    return b.correct - a.correct
                } else if (sortBy === 'accuracy') {
                    return (b.confidence || 0) - (a.confidence || 0)
                }
                return 0
            })
            setRaces(sorted)
        } else {
            setRaces([])
        }
        setLoading(false)
    }, [raceHistory, sortBy])

    const calculateModelAccuracy = () => {
        if (races.length === 0) return 0
        const correctCount = races.filter(r => r.correct).length
        return Math.round((correctCount / races.length) * 100)
    }

    const getResultColor = (correct) => {
        return correct ? '#22c55e' : '#ef4444'
    }

    const getConfidenceColor = (confidence) => {
        if (confidence >= 70) return '#22c55e'
        if (confidence >= 50) return '#eab308'
        return '#ef4444'
    }

    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '16px',
            overflow: 'hidden'
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                justifyContent: 'flex-start',
                alignItems: 'center',
                marginBottom: '16px'
            }}>
                <div style={{
                    fontSize: '13px',
                    fontWeight: '700',
                    letterSpacing: '0.5px',
                    color: '#888888',
                    textTransform: 'uppercase'
                }}>
                    RACE HISTORY (LAST 5)
                </div>
            </div>

            {/* Controls */}
            <div style={{
                display: 'flex',
                gap: '8px',
                marginBottom: '12px',
                flexWrap: 'wrap'
            }}>
                <button
                    onClick={() => setSortBy('date')}
                    type="button"
                    className="ui-chip ui-focus-ring"
                    aria-pressed={sortBy === 'date'}
                    style={{
                        '--chip-color': '#00d4ff',
                        backgroundColor: sortBy === 'date' ? 'rgba(0, 212, 255, 0.2)' : undefined
                    }}
                >
                    By Date
                </button>
                <button
                    onClick={() => setSortBy('result')}
                    type="button"
                    className="ui-chip ui-focus-ring"
                    aria-pressed={sortBy === 'result'}
                    style={{
                        '--chip-color': '#22c55e',
                        backgroundColor: sortBy === 'result' ? 'rgba(34, 197, 94, 0.2)' : undefined
                    }}
                >
                    By Result
                </button>
                <button
                    onClick={() => setSortBy('accuracy')}
                    type="button"
                    className="ui-chip ui-focus-ring"
                    aria-pressed={sortBy === 'accuracy'}
                    style={{
                        '--chip-color': '#ea580c',
                        backgroundColor: sortBy === 'accuracy' ? 'rgba(234, 88, 12, 0.2)' : undefined
                    }}
                >
                    By Confidence
                </button>
            </div>

            {/* Table */}
            {loading ? (
                <div style={{
                    textAlign: 'center',
                    padding: '40px',
                    color: '#888888',
                    fontSize: '12px'
                }}>
                    Loading race history...
                </div>
            ) : races.length === 0 ? (
                <div style={{
                    textAlign: 'center',
                    padding: '40px',
                    color: '#888888',
                    fontSize: '12px'
                }}>
                    No race history available yet
                </div>
            ) : (
                <div style={{
                    overflowX: 'auto'
                }}>
                    <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '13px'
                    }}>
                        <thead>
                            <tr style={{
                                borderBottom: '1px solid #3a3a3a',
                                backgroundColor: 'rgba(80, 80, 80, 0.1)'
                            }}>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'left',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    RACE
                                </th>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    DATE
                                </th>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    PREDICTED
                                </th>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    ACTUAL
                                </th>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    RESULT
                                </th>
                                <th scope="col" style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    CONFIDENCE
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {races.map((race, idx) => (
                                <tr
                                    key={idx}
                                    className="ui-row-hover"
                                    style={{
                                        borderBottom: '1px solid rgba(80, 80, 80, 0.3)',
                                        cursor: 'pointer'
                                    }}
                                >
                                    <td style={{
                                        padding: '10px 8px',
                                        color: '#cbd5e1',
                                        fontWeight: '600'
                                    }}>
                                        {race.race}
                                    </td>
                                    <td style={{
                                        padding: '10px 8px',
                                        textAlign: 'center',
                                        color: '#888888',
                                        fontSize: '12px'
                                    }}>
                                        {race.date ? new Date(race.date).toLocaleDateString() : 'N/A'}
                                    </td>
                                    <td style={{
                                        padding: '10px 8px',
                                        textAlign: 'center',
                                        color: '#00d4ff',
                                        fontWeight: '600'
                                    }}>
                                        {race.predicted_winner || 'N/A'}
                                    </td>
                                    <td style={{
                                        padding: '10px 8px',
                                        textAlign: 'center',
                                        color: '#ea580c',
                                        fontWeight: '600'
                                    }}>
                                        {race.actual_winner || 'N/A'}
                                    </td>
                                    <td style={{
                                        padding: '10px 8px',
                                        textAlign: 'center'
                                    }}>
                                        <span style={{
                                            color: getResultColor(race.correct),
                                            fontWeight: '700',
                                            fontSize: '11px',
                                            padding: '4px 8px',
                                            backgroundColor: getResultColor(race.correct) === '#22c55e' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                            borderRadius: '3px'
                                        }}>
                                            {race.correct ? '✓ CORRECT' : '✗ MISSED'}
                                        </span>
                                    </td>
                                    <td style={{
                                        padding: '10px 8px',
                                        textAlign: 'center',
                                        color: getConfidenceColor(race.confidence || 0),
                                        fontWeight: '600'
                                    }}>
                                        {race.confidence || 0}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
