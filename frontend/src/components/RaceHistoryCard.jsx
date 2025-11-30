import React, { useState, useEffect } from 'react'

export default function RaceHistoryCard({ raceHistory = [] }) {
    const [races, setRaces] = useState([])
    const [sortBy, setSortBy] = useState('date')
    const [hoveredRow, setHoveredRow] = useState(null)
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
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px'
            }}>
                <div style={{
                    fontSize: '11px',
                    fontWeight: '700',
                    letterSpacing: '1px',
                    color: '#888888',
                    textTransform: 'uppercase'
                }}>
                    RACE HISTORY (LAST 5)
                </div>
                <div style={{
                    fontSize: '12px',
                    fontWeight: '700',
                    color: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    padding: '4px 12px',
                    borderRadius: '4px'
                }}>
                    Accuracy: {calculateModelAccuracy()}%
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
                    style={{
                        padding: '6px 12px',
                        backgroundColor: sortBy === 'date' ? 'rgba(0, 212, 255, 0.2)' : 'rgba(80, 80, 80, 0.2)',
                        border: `1px solid ${sortBy === 'date' ? '#00d4ff' : '#3a3a3a'}`,
                        borderRadius: '4px',
                        color: sortBy === 'date' ? '#00d4ff' : '#888888',
                        fontSize: '9px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                        e.target.style.borderColor = '#00d4ff'
                        e.target.style.color = '#00d4ff'
                    }}
                    onMouseLeave={(e) => {
                        if (sortBy !== 'date') {
                            e.target.style.borderColor = '#3a3a3a'
                            e.target.style.color = '#888888'
                        }
                    }}
                >
                    By Date
                </button>
                <button
                    onClick={() => setSortBy('result')}
                    style={{
                        padding: '6px 12px',
                        backgroundColor: sortBy === 'result' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(80, 80, 80, 0.2)',
                        border: `1px solid ${sortBy === 'result' ? '#22c55e' : '#3a3a3a'}`,
                        borderRadius: '4px',
                        color: sortBy === 'result' ? '#22c55e' : '#888888',
                        fontSize: '9px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                        e.target.style.borderColor = '#22c55e'
                        e.target.style.color = '#22c55e'
                    }}
                    onMouseLeave={(e) => {
                        if (sortBy !== 'result') {
                            e.target.style.borderColor = '#3a3a3a'
                            e.target.style.color = '#888888'
                        }
                    }}
                >
                    By Result
                </button>
                <button
                    onClick={() => setSortBy('accuracy')}
                    style={{
                        padding: '6px 12px',
                        backgroundColor: sortBy === 'accuracy' ? 'rgba(234, 88, 12, 0.2)' : 'rgba(80, 80, 80, 0.2)',
                        border: `1px solid ${sortBy === 'accuracy' ? '#ea580c' : '#3a3a3a'}`,
                        borderRadius: '4px',
                        color: sortBy === 'accuracy' ? '#ea580c' : '#888888',
                        fontSize: '9px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                        e.target.style.borderColor = '#ea580c'
                        e.target.style.color = '#ea580c'
                    }}
                    onMouseLeave={(e) => {
                        if (sortBy !== 'accuracy') {
                            e.target.style.borderColor = '#3a3a3a'
                            e.target.style.color = '#888888'
                        }
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
                        fontSize: '10px'
                    }}>
                        <thead>
                            <tr style={{
                                borderBottom: '1px solid #3a3a3a',
                                backgroundColor: 'rgba(80, 80, 80, 0.1)'
                            }}>
                                <th style={{
                                    padding: '10px 8px',
                                    textAlign: 'left',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    RACE
                                </th>
                                <th style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    DATE
                                </th>
                                <th style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    PREDICTED
                                </th>
                                <th style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    ACTUAL
                                </th>
                                <th style={{
                                    padding: '10px 8px',
                                    textAlign: 'center',
                                    color: '#888888',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px'
                                }}>
                                    RESULT
                                </th>
                                <th style={{
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
                                    style={{
                                        borderBottom: '1px solid rgba(80, 80, 80, 0.3)',
                                        backgroundColor: hoveredRow === idx ? 'rgba(0, 212, 255, 0.08)' : 'transparent',
                                        transition: 'background-color 0.2s ease',
                                        cursor: 'pointer'
                                    }}
                                    onMouseEnter={() => setHoveredRow(idx)}
                                    onMouseLeave={() => setHoveredRow(null)}
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
                                        fontSize: '9px'
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
                                            fontSize: '9px',
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
