import React from 'react'

export default function Header({ activeTab, setActiveTab }) {
    const tabs = [
        { id: 'current', label: 'CURRENT RACE' },
        { id: 'standings', label: 'DRIVER STANDINGS' },
        { id: 'constructor', label: 'CONSTRUCTOR STANDINGS' },
        { id: 'circuit', label: 'CIRCUIT MAP' },
        { id: 'matchup', label: 'HEAD-TO-HEAD' },
        { id: 'mlops', label: 'MODEL MONITOR' }
    ]

    return (
        <header style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
            marginBottom: '28px',
            borderBottom: '1px solid #3a3a3a',
            backgroundColor: 'rgba(0, 0, 0, 0.4)',
            padding: '16px 24px',
            borderRadius: '8px'
        }}>
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '32px',
                flexWrap: 'wrap'
            }}>
                {/* Logo */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    cursor: 'pointer',
                    transition: 'transform 0.3s ease'
                }}>
                    <div style={{
                        fontSize: '24px',
                        fontWeight: '800',
                        letterSpacing: '2px',
                        color: '#fff',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <span>🏁</span>
                        <span>F1 PREDICT</span>
                    </div>
                </div>

                {/* Navigation */}
                <nav style={{
                    display: 'flex',
                    gap: '8px',
                    flexWrap: 'wrap'
                }}>
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: activeTab === tab.id ? '#00d4ff' : '#888888',
                                fontSize: '13px',
                                fontWeight: '700',
                                letterSpacing: '0.5px',
                                cursor: 'pointer',
                                padding: '6px 10px',
                                borderBottom: activeTab === tab.id ? '2px solid #00d4ff' : '2px solid transparent',
                                transition: 'all 0.3s ease',
                                whiteSpace: 'nowrap'
                            }}
                            onMouseEnter={(e) => {
                                if (activeTab !== tab.id) e.target.style.color = '#cbd5e1'
                            }}
                            onMouseLeave={(e) => {
                                if (activeTab !== tab.id) e.target.style.color = '#888888'
                            }}
                        >
                            {tab.label}
                        </button>
                    ))}
                </nav>
            </div>

            <div style={{
                color: '#888888',
                fontSize: '13px',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
            }}>
                <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#10b981',
                    animation: 'pulse 2s infinite'
                }} />
                LIVE
            </div>
        </header>
    )
}
