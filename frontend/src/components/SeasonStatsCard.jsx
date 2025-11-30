import React, { useState } from 'react'

export default function SeasonStatsCard({
    races = ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Baku'],
    raceDetails = [],
    correctPredictions = 3,
    totalRaces = 5,
    accuracy = 60
}) {
    const [hoveredSection, setHoveredSection] = useState(null)
    const [selectedSection, setSelectedSection] = useState(null)
    const [hoveredRace, setHoveredRace] = useState(null)

    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '16px',
            transition: 'all 0.3s ease'
        }}>
            <div style={{
                fontSize: '11px',
                fontWeight: '700',
                letterSpacing: '1px',
                color: '#888888',
                textTransform: 'uppercase',
                marginBottom: '8px'
            }}>
                SEASON
            </div>
            <div style={{
                fontSize: '10px',
                color: '#888888',
                marginBottom: '12px',
                fontWeight: '600'
            }}>
                PREDICTED VS. ACTUAL WINS
            </div>

            {/* Stats Numbers */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                marginBottom: '12px'
            }}>
                <div
                    style={{
                        backgroundColor: hoveredSection === 'predicted' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(34, 197, 94, 0.1)',
                        padding: '10px',
                        borderRadius: '6px',
                        borderLeft: '3px solid #22c55e',
                        cursor: 'pointer',
                        transform: hoveredSection === 'predicted' ? 'scale(1.05)' : 'scale(1)',
                        transition: 'all 0.3s ease',
                        border: selectedSection === 'predicted' ? '1px solid #22c55e' : 'none'
                    }}
                    onMouseEnter={() => setHoveredSection('predicted')}
                    onMouseLeave={() => setHoveredSection(null)}
                    onClick={() => setSelectedSection(selectedSection === 'predicted' ? null : 'predicted')}
                >
                    <div style={{
                        fontSize: '9px',
                        color: '#888888',
                        marginBottom: '2px'
                    }}>
                        CORRECT PREDICTIONS
                    </div>
                    <div style={{
                        fontSize: '16px',
                        fontWeight: '800',
                        color: '#22c55e'
                    }}>
                        {correctPredictions}
                    </div>
                    {selectedSection === 'predicted' && (
                        <div style={{
                            fontSize: '8px',
                            color: '#22c55e',
                            marginTop: '4px'
                        }}>
                            ML Model Got It Right
                        </div>
                    )}
                </div>
                <div
                    style={{
                        backgroundColor: hoveredSection === 'actual' ? 'rgba(234, 88, 12, 0.2)' : 'rgba(234, 88, 12, 0.1)',
                        padding: '10px',
                        borderRadius: '6px',
                        borderLeft: '3px solid #ea580c',
                        cursor: 'pointer',
                        transform: hoveredSection === 'actual' ? 'scale(1.05)' : 'scale(1)',
                        transition: 'all 0.3s ease',
                        border: selectedSection === 'actual' ? '1px solid #ea580c' : 'none'
                    }}
                    onMouseEnter={() => setHoveredSection('actual')}
                    onMouseLeave={() => setHoveredSection(null)}
                    onClick={() => setSelectedSection(selectedSection === 'actual' ? null : 'actual')}
                >
                    <div style={{
                        fontSize: '9px',
                        color: '#888888',
                        marginBottom: '2px'
                    }}>
                        TOTAL RACES
                    </div>
                    <div style={{
                        fontSize: '16px',
                        fontWeight: '800',
                        color: '#ea580c'
                    }}>
                        {totalRaces}
                    </div>
                    {selectedSection === 'actual' && (
                        <div style={{
                            fontSize: '8px',
                            color: '#ea580c',
                            marginTop: '4px'
                        }}>
                            Races Analyzed
                        </div>
                    )}
                </div>
            </div>

            {/* Interactive Progress Bar */}
            <div style={{
                display: 'flex',
                height: '8px',
                backgroundColor: 'rgba(80, 80, 80, 0.5)',
                borderRadius: '4px',
                overflow: 'hidden',
                marginBottom: '14px',
                gap: '1px',
                cursor: 'pointer'
            }}>
                <div
                    style={{
                        flex: correctPredictions,
                        backgroundColor: hoveredSection === 'predicted' ? '#2dd96f' : '#22c55e',
                        borderRadius: '2px',
                        transition: 'all 0.3s ease',
                        position: 'relative'
                    }}
                    onMouseEnter={() => setHoveredSection('predicted')}
                    onMouseLeave={() => setHoveredSection(null)}
                    onClick={() => setSelectedSection(selectedSection === 'predicted' ? null : 'predicted')}
                >
                    {hoveredSection === 'predicted' && (
                        <div style={{
                            position: 'absolute',
                            top: '-25px',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            backgroundColor: '#22c55e',
                            color: '#1a1a1a',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: '700',
                            whiteSpace: 'nowrap',
                            zIndex: 10
                        }}>
                            {correctPredictions} Correct
                        </div>
                    )}
                </div>
                <div
                    style={{
                        flex: totalRaces - correctPredictions,
                        backgroundColor: hoveredSection === 'actual' ? '#ff6a1c' : '#ea580c',
                        borderRadius: '2px',
                        transition: 'all 0.3s ease',
                        position: 'relative'
                    }}
                    onMouseEnter={() => setHoveredSection('actual')}
                    onMouseLeave={() => setHoveredSection(null)}
                    onClick={() => setSelectedSection(selectedSection === 'actual' ? null : 'actual')}
                >
                    {hoveredSection === 'actual' && (
                        <div style={{
                            position: 'absolute',
                            top: '-25px',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            backgroundColor: '#ea580c',
                            color: '#ffffff',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: '700',
                            whiteSpace: 'nowrap',
                            zIndex: 10
                        }}>
                            {totalRaces - correctPredictions} Missed
                        </div>
                    )}
                </div>
            </div>

            {/* Race Results with Driver Names */}
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                marginBottom: '12px',
                maxHeight: '200px',
                overflowY: 'auto'
            }}>
                {raceDetails.length > 0 ? raceDetails.map((detail, idx) => (
                    <div
                        key={idx}
                        style={{
                            padding: '8px',
                            backgroundColor: hoveredRace === idx ? 'rgba(0, 212, 255, 0.1)' : 'rgba(50, 50, 50, 0.3)',
                            borderRadius: '6px',
                            borderLeft: `3px solid ${detail.correct ? '#22c55e' : '#ef4444'}`,
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                            fontSize: '9px'
                        }}
                        onMouseEnter={() => setHoveredRace(idx)}
                        onMouseLeave={() => setHoveredRace(null)}
                    >
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '4px'
                        }}>
                            <span style={{
                                color: '#cbd5e1',
                                fontWeight: '700',
                                fontSize: '10px'
                            }}>{detail.race}</span>
                            <span style={{
                                color: detail.correct ? '#22c55e' : '#ef4444',
                                fontWeight: '700',
                                fontSize: '8px'
                            }}>
                                {detail.correct ? '✓ CORRECT' : '✗ MISSED'}
                            </span>
                        </div>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '8px',
                            color: '#888888'
                        }}>
                            <div>
                                <span style={{ color: '#64748b' }}>Predicted: </span>
                                <span style={{ color: '#00d4ff', fontWeight: '600' }}>{detail.predicted_winner}</span>
                            </div>
                            <div>
                                <span style={{ color: '#64748b' }}>Actual: </span>
                                <span style={{ color: '#ea580c', fontWeight: '600' }}>{detail.actual_winner}</span>
                            </div>
                        </div>
                        {hoveredRace === idx && detail.confidence && (
                            <div style={{
                                marginTop: '4px',
                                color: '#64748b',
                                fontSize: '8px'
                            }}>
                                Confidence: {detail.confidence}%
                            </div>
                        )}
                    </div>
                )) : (
                    <div style={{
                        textAlign: 'center',
                        padding: '12px',
                        color: '#888888',
                        fontSize: '9px'
                    }}>
                        No race data available
                    </div>
                )}
            </div>

            {/* Accuracy */}
            <div
                style={{
                    padding: '8px',
                    backgroundColor: hoveredSection === 'accuracy' ? 'rgba(0, 212, 255, 0.15)' : 'rgba(0, 212, 255, 0.1)',
                    borderRadius: '6px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '10px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    border: selectedSection === 'accuracy' ? '1px solid #00d4ff' : 'none'
                }}
                onMouseEnter={() => setHoveredSection('accuracy')}
                onMouseLeave={() => setHoveredSection(null)}
                onClick={() => setSelectedSection(selectedSection === 'accuracy' ? null : 'accuracy')}
            >
                <span style={{ color: '#888888' }}>Model Accuracy:</span>
                <span style={{ fontWeight: '700', color: '#00d4ff' }}>{accuracy}%</span>
            </div>

            {selectedSection === 'accuracy' && (
                <div style={{
                    marginTop: '8px',
                    padding: '8px',
                    backgroundColor: 'rgba(0, 212, 255, 0.05)',
                    borderRadius: '6px',
                    fontSize: '9px',
                    color: '#cbd5e1',
                    lineHeight: '1.6'
                }}>
                    Based on last {totalRaces} races. {correctPredictions} correct predictions out of {totalRaces} races analyzed from historical data.
                </div>
            )}
        </div>
    )
}
