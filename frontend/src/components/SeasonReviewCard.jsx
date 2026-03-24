import React, { useState } from 'react'

/**
 * SeasonReviewCard - Displays full season prediction accuracy review
 * Shows how the ML model performed across an entire season:
 * - Overall accuracy stats (wins, podiums)
 * - Race-by-race breakdown with predicted vs actual winners
 * - Visual indicators for correct/incorrect predictions
 */
export default function SeasonReviewCard({ seasonReview, onYearChange }) {
    const [sortBy, setSortBy] = useState('round') // round, result, confidence
    const [hoveredRow, setHoveredRow] = useState(null)

    if (!seasonReview || !seasonReview.races || seasonReview.races.length === 0) {
        return (
            <div style={{
                backgroundColor: '#1a1f2e',
                border: '1px solid #334155',
                borderRadius: '12px',
                padding: '32px',
                textAlign: 'center'
            }}>
                <div style={{ fontSize: '14px', color: '#94a3b8' }}>
                    No season review data available
                </div>
            </div>
        )
    }

    const { year, races, stats, availableYears } = seasonReview

    // Sort races
    const sortedRaces = [...races].sort((a, b) => {
        if (sortBy === 'result') {
            return (b.correct ? 1 : 0) - (a.correct ? 1 : 0)
        }
        if (sortBy === 'confidence') {
            return (b.confidence || 0) - (a.confidence || 0)
        }
        return a.round - b.round // chronological
    })

    const getResultColor = (correct) => correct ? '#22c55e' : '#ef4444'
    const getConfidenceColor = (conf) => {
        if (conf >= 70) return '#22c55e'
        if (conf >= 50) return '#f59e0b'
        return '#ef4444'
    }

    return (
        <div style={{
            backgroundColor: '#1a1f2e',
            border: '1px solid #334155',
            borderRadius: '12px',
            overflow: 'hidden'
        }}>
            {/* Header with Season Stats */}
            <div style={{
                background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
                padding: '24px',
                borderBottom: '1px solid #334155'
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: '20px'
                }}>
                    <div>
                        <div style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '1.5px',
                            color: '#f59e0b',
                            textTransform: 'uppercase',
                            marginBottom: '6px'
                        }}>
                            SEASON REVIEW
                        </div>
                        <div style={{
                            fontSize: '24px',
                            fontWeight: '800',
                            color: '#e2e8f0',
                            letterSpacing: '-0.5px'
                        }}>
                            {year} F1 Season
                        </div>
                        <div style={{
                            fontSize: '14px',
                            color: '#94a3b8',
                            marginTop: '4px'
                        }}>
                            Model Prediction Accuracy Report
                        </div>
                    </div>
                    {/* Year Selector */}
                    {availableYears && availableYears.length > 1 && (
                        <select
                            value={year}
                            onChange={(e) => onYearChange && onYearChange(parseInt(e.target.value))}
                            style={{
                                backgroundColor: '#0f172a',
                                color: '#e2e8f0',
                                border: '1px solid #475569',
                                borderRadius: '6px',
                                padding: '6px 12px',
                                fontSize: '14px',
                                fontWeight: '600',
                                cursor: 'pointer',
                                outline: 'none'
                            }}
                        >
                            {availableYears.map(y => (
                                <option key={y} value={y}>{y}</option>
                            ))}
                        </select>
                    )}
                </div>

                {/* Stats Grid */}
                <div className="stats-grid-4">
                    {/* Winner Accuracy */}
                    <div style={{
                        backgroundColor: 'rgba(0, 212, 255, 0.08)',
                        border: '1px solid rgba(0, 212, 255, 0.2)',
                        borderRadius: '10px',
                        padding: '16px',
                        textAlign: 'center'
                    }}>
                        <div style={{
                            fontSize: '28px',
                            fontWeight: '800',
                            color: '#00d4ff',
                            lineHeight: 1
                        }}>
                            {stats.accuracy_percentage || 0}%
                        </div>
                        <div style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '0.5px',
                            color: '#94a3b8',
                            marginTop: '8px',
                            textTransform: 'uppercase'
                        }}>
                            Winner Accuracy
                        </div>
                    </div>

                    {/* Correct Predictions */}
                    <div style={{
                        backgroundColor: 'rgba(34, 197, 94, 0.08)',
                        border: '1px solid rgba(34, 197, 94, 0.2)',
                        borderRadius: '10px',
                        padding: '16px',
                        textAlign: 'center'
                    }}>
                        <div style={{
                            fontSize: '28px',
                            fontWeight: '800',
                            color: '#22c55e',
                            lineHeight: 1
                        }}>
                            {stats.correct_predictions || 0}/{stats.total_races || 0}
                        </div>
                        <div style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '0.5px',
                            color: '#94a3b8',
                            marginTop: '8px',
                            textTransform: 'uppercase'
                        }}>
                            Correct Winners
                        </div>
                    </div>

                    {/* Podium Accuracy */}
                    <div style={{
                        backgroundColor: 'rgba(245, 158, 11, 0.08)',
                        border: '1px solid rgba(245, 158, 11, 0.2)',
                        borderRadius: '10px',
                        padding: '16px',
                        textAlign: 'center'
                    }}>
                        <div style={{
                            fontSize: '28px',
                            fontWeight: '800',
                            color: '#f59e0b',
                            lineHeight: 1
                        }}>
                            {stats.podium_accuracy_percentage || 0}%
                        </div>
                        <div style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '0.5px',
                            color: '#94a3b8',
                            marginTop: '8px',
                            textTransform: 'uppercase'
                        }}>
                            Podium Accuracy
                        </div>
                    </div>

                    {/* Total Races */}
                    <div style={{
                        backgroundColor: 'rgba(148, 163, 184, 0.08)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: '10px',
                        padding: '16px',
                        textAlign: 'center'
                    }}>
                        <div style={{
                            fontSize: '28px',
                            fontWeight: '800',
                            color: '#e2e8f0',
                            lineHeight: 1
                        }}>
                            {stats.total_with_data || stats.total_races || 0}
                        </div>
                        <div style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '0.5px',
                            color: '#94a3b8',
                            marginTop: '8px',
                            textTransform: 'uppercase'
                        }}>
                            Races Analyzed
                        </div>
                    </div>
                </div>
            </div>

            {/* Sort Controls */}
            <div style={{
                display: 'flex',
                gap: '8px',
                padding: '12px 24px',
                borderBottom: '1px solid rgba(80, 80, 80, 0.3)',
                backgroundColor: 'rgba(15, 23, 42, 0.5)'
            }}>
                <span style={{
                    fontSize: '11px',
                    fontWeight: '700',
                    letterSpacing: '0.5px',
                    color: '#64748b',
                    alignSelf: 'center',
                    marginRight: '8px'
                }}>
                    SORT BY:
                </span>
                {[
                    { key: 'round', label: 'ROUND' },
                    { key: 'result', label: 'RESULT' },
                    { key: 'confidence', label: 'CONFIDENCE' }
                ].map(({ key, label }) => (
                    <button
                        key={key}
                        onClick={() => setSortBy(key)}
                        style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            letterSpacing: '0.5px',
                            color: sortBy === key ? '#00d4ff' : '#64748b',
                            backgroundColor: sortBy === key ? 'rgba(0, 212, 255, 0.1)' : 'transparent',
                            border: `1px solid ${sortBy === key ? 'rgba(0, 212, 255, 0.3)' : 'transparent'}`,
                            borderRadius: '4px',
                            padding: '4px 10px',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {/* Race Results Table */}
            <div style={{
                overflowX: 'auto',
                maxHeight: '500px',
                overflowY: 'auto'
            }}>
                <table style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '13px'
                }}>
                    <thead>
                        <tr style={{
                            backgroundColor: 'rgba(15, 23, 42, 0.8)',
                            position: 'sticky',
                            top: 0,
                            zIndex: 1
                        }}>
                            <th style={thStyle}>RD</th>
                            <th style={{ ...thStyle, textAlign: 'left' }}>RACE</th>
                            <th style={thStyle}>DATE</th>
                            <th style={thStyle}>PREDICTED</th>
                            <th style={thStyle}>ACTUAL</th>
                            <th style={thStyle}>RESULT</th>
                            <th style={thStyle}>CONFIDENCE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedRaces.map((race, idx) => (
                            <tr
                                key={idx}
                                style={{
                                    borderBottom: '1px solid rgba(80, 80, 80, 0.2)',
                                    backgroundColor: hoveredRow === idx
                                        ? 'rgba(0, 212, 255, 0.06)'
                                        : race.correct
                                            ? 'rgba(34, 197, 94, 0.03)'
                                            : 'transparent',
                                    transition: 'background-color 0.2s ease',
                                    cursor: 'pointer'
                                }}
                                onMouseEnter={() => setHoveredRow(idx)}
                                onMouseLeave={() => setHoveredRow(null)}
                            >
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center',
                                    color: '#64748b',
                                    fontWeight: '700',
                                    fontSize: '12px'
                                }}>
                                    R{race.round}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    color: '#e2e8f0',
                                    fontWeight: '600',
                                    fontSize: '13px',
                                    maxWidth: '200px',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    whiteSpace: 'nowrap'
                                }}>
                                    {race.race}
                                    {race.podium_correct && !race.correct && (
                                        <span style={{
                                            fontSize: '11px',
                                            color: '#f59e0b',
                                            marginLeft: '6px',
                                            fontWeight: '600'
                                        }}>
                                            PODIUM
                                        </span>
                                    )}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center',
                                    color: '#64748b',
                                    fontSize: '12px'
                                }}>
                                    {race.date ? new Date(race.date).toLocaleDateString('en-US', {
                                        month: 'short',
                                        day: 'numeric'
                                    }) : '-'}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center',
                                    color: '#00d4ff',
                                    fontWeight: '700',
                                    fontSize: '13px'
                                }}>
                                    {race.predicted_winner || 'N/A'}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center',
                                    color: '#ea580c',
                                    fontWeight: '700',
                                    fontSize: '13px'
                                }}>
                                    {race.actual_winner || 'N/A'}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center'
                                }}>
                                    {race.status === 'no_qualifying_data' ? (
                                        <span style={{
                                            color: '#64748b',
                                            fontWeight: '600',
                                            fontSize: '11px',
                                            padding: '3px 8px',
                                            backgroundColor: 'rgba(100, 116, 139, 0.1)',
                                            borderRadius: '3px'
                                        }}>
                                            NO DATA
                                        </span>
                                    ) : (
                                        <span style={{
                                            color: getResultColor(race.correct),
                                            fontWeight: '700',
                                            fontSize: '11px',
                                            padding: '3px 8px',
                                            backgroundColor: race.correct
                                                ? 'rgba(34, 197, 94, 0.12)'
                                                : 'rgba(239, 68, 68, 0.12)',
                                            borderRadius: '3px',
                                            letterSpacing: '0.5px'
                                        }}>
                                            {race.correct ? '✓ HIT' : '✗ MISS'}
                                        </span>
                                    )}
                                </td>
                                <td style={{
                                    padding: '12px 10px',
                                    textAlign: 'center'
                                }}>
                                    <span style={{
                                        color: getConfidenceColor(race.confidence || 0),
                                        fontWeight: '700',
                                        fontSize: '13px'
                                    }}>
                                        {race.confidence || 0}%
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer Summary */}
            <div style={{
                padding: '16px 24px',
                borderTop: '1px solid #334155',
                backgroundColor: 'rgba(15, 23, 42, 0.5)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div style={{
                    fontSize: '12px',
                    color: '#94a3b8',
                    fontWeight: '600'
                }}>
                    Model: XGBoost v3 (Hybrid: 60% ML + 40% Qualifying)
                </div>
                <div style={{
                    display: 'flex',
                    gap: '16px',
                    fontSize: '12px',
                    fontWeight: '600'
                }}>
                    <span style={{ color: '#22c55e' }}>
                        ✓ {stats.correct_predictions || 0} correct
                    </span>
                    <span style={{ color: '#ef4444' }}>
                        ✗ {(stats.total_races || 0) - (stats.correct_predictions || 0)} missed
                    </span>
                    <span style={{ color: '#f59e0b' }}>
                        ~ {stats.podium_correct || 0} podium hits
                    </span>
                </div>
            </div>
        </div>
    )
}

// Shared table header style
const thStyle = {
    padding: '10px',
    textAlign: 'center',
    color: '#64748b',
    fontWeight: '700',
    letterSpacing: '0.5px',
    fontSize: '11px',
    textTransform: 'uppercase'
}
