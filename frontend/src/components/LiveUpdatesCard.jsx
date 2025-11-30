import React, { useState, useEffect } from 'react'

export default function LiveUpdatesCard({ liveUpdates }) {
    const [updates, setUpdates] = useState([])
    const [isLive, setIsLive] = useState(false)
    const [animatingIndex, setAnimatingIndex] = useState(0)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (!liveUpdates) {
            setLoading(true)
            return
        }

        try {
            setIsLive(liveUpdates.isLive)

            if (liveUpdates.positions && liveUpdates.positions.length > 0) {
                // Format position data into updates
                const formatted = liveUpdates.positions.slice(-4).map(pos =>
                    `P${pos.position || '?'}: ${pos.driver_number} - ${new Date(pos.date).toLocaleTimeString()}`
                )
                setUpdates(formatted)
            } else if (liveUpdates.session) {
                // Show session info if no live positions
                setUpdates([
                    `${liveUpdates.session.session_name} - ${liveUpdates.session.meeting_name}`,
                    `Circuit: ${liveUpdates.session.circuit_short_name}`,
                    `Date: ${new Date(liveUpdates.session.date_start).toLocaleDateString()}`,
                    liveUpdates.isLive ? 'Race in progress...' : 'Race completed'
                ])
            } else {
                setUpdates([
                    'No live race data available',
                    'Check back during race weekend',
                    'Stay tuned for updates',
                    'Next race coming soon'
                ])
            }
        } catch (error) {
            console.error('Failed to process live updates:', error)
            setUpdates([
                'Unable to fetch live updates',
                'Check back during race weekend',
                'Data will appear when race is active',
                'Thank you for your patience'
            ])
        } finally {
            setLoading(false)
        }
    }, [liveUpdates])

    useEffect(() => {
        if (updates.length > 0) {
            const interval = setInterval(() => {
                setAnimatingIndex(prev => (prev + 1) % updates.length)
            }, 4000)
            return () => clearInterval(interval)
        }
    }, [updates.length])

    return (
        <div style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #3a3a3a',
            borderRadius: '10px',
            padding: '16px',
            overflow: 'hidden'
        }}>
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '12px'
            }}>
                <div style={{
                    fontSize: '11px',
                    fontWeight: '700',
                    letterSpacing: '1px',
                    color: '#888888',
                    textTransform: 'uppercase',
                    flex: 1
                }}>
                    LIVE RACE UPDATES
                </div>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    color: '#ef4444',
                    fontSize: '10px',
                    fontWeight: '700',
                    letterSpacing: '1px'
                }}>
                    <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: isLive ? '#ef4444' : '#64748b',
                        animation: isLive ? 'pulse 1s infinite' : 'none'
                    }} />
                    {isLive ? 'LIVE' : 'RECENT'}
                </div>
            </div>

            {/* Updates List */}
            {loading ? (
                <div style={{
                    textAlign: 'center',
                    padding: '40px',
                    color: '#888888',
                    fontSize: '12px'
                }}>
                    Loading updates...
                </div>
            ) : (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    maxHeight: '240px',
                    overflowY: 'auto',
                    paddingRight: '8px'
                }}>
                    {updates.map((update, idx) => (
                        <div
                            key={idx}
                            style={{
                                padding: '12px',
                                backgroundColor: idx === animatingIndex ? 'rgba(0, 212, 255, 0.15)' : 'rgba(80, 80, 80, 0.3)',
                                border: `1px solid ${idx === animatingIndex ? 'rgba(0, 212, 255, 0.4)' : 'rgba(80, 80, 80, 0.4)'}`,
                                borderRadius: '6px',
                                fontSize: '11px',
                                color: idx === animatingIndex ? '#cbd5e1' : '#888888',
                                lineHeight: '1.5',
                                transition: 'all 0.3s ease',
                                cursor: 'pointer',
                                borderLeft: `2px solid ${idx === animatingIndex ? '#00d4ff' : 'transparent'}`
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = 'rgba(0, 212, 255, 0.2)'
                                e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.5)'
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = idx === animatingIndex ? 'rgba(0, 212, 255, 0.15)' : 'rgba(80, 80, 80, 0.3)'
                                e.currentTarget.style.borderColor = idx === animatingIndex ? 'rgba(0, 212, 255, 0.4)' : 'rgba(80, 80, 80, 0.4)'
                            }}
                        >
                            {update}
                        </div>
                    ))}
                </div>
            )}

            <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
        </div>
    )
}
