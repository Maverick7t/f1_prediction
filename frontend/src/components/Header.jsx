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
            marginBottom: '32px',
            borderBottom: '1px solid #3a3a3a',
            paddingBottom: '20px',
            backgroundColor: 'rgba(0, 0, 0, 0.4)',
            padding: '20px',
            marginLeft: '-20px',
            marginRight: '-20px',
            paddingLeft: '20px',
            paddingRight: '20px',
            borderRadius: '8px'
        }}>
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '40px'
            }}>
                {/* Logo with checkered flag pattern */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    cursor: 'pointer',
                    transition: 'transform 0.3s ease',
                    hover: { transform: 'scale(1.05)' }
                }}>
                    <div style={{
                        fontSize: '28px',
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
                    gap: '24px'
                }}>
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: activeTab === tab.id ? '#00d4ff' : '#888888',
                                fontSize: '11px',
                                fontWeight: '700',
                                letterSpacing: '1px',
                                cursor: 'pointer',
                                paddingBottom: activeTab === tab.id ? '6px' : '0',
                                borderBottom: activeTab === tab.id ? '2px solid #00d4ff' : 'none',
                                transition: 'all 0.3s ease'
                            }}
                            onMouseEnter={(e) => !activeTab === tab.id && (e.target.style.color = '#94a3b8')}
                            onMouseLeave={(e) => !activeTab === tab.id && (e.target.style.color = '#64748b')}
                        >
                            {tab.label}
                        </button>
                    ))}
                </nav>
            </div>

            <div style={{
                color: '#888888',
                fontSize: '11px',
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
