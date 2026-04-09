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

    const tabButtonRefs = React.useRef(new Map())

    const focusTab = React.useCallback((tabId) => {
        const el = tabButtonRefs.current.get(tabId)
        if (el) el.focus()
    }, [])

    const handleTabKeyDown = React.useCallback((e, tabId) => {
        const currentIndex = tabs.findIndex(t => t.id === tabId)
        if (currentIndex < 0) return

        let nextIndex = null
        if (e.key === 'ArrowRight') nextIndex = (currentIndex + 1) % tabs.length
        if (e.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + tabs.length) % tabs.length
        if (e.key === 'Home') nextIndex = 0
        if (e.key === 'End') nextIndex = tabs.length - 1

        if (nextIndex === null) return
        e.preventDefault()

        const nextId = tabs[nextIndex].id
        setActiveTab(nextId)
        requestAnimationFrame(() => focusTab(nextId))
    }, [focusTab, setActiveTab, tabs])

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
                <nav
                    role="tablist"
                    aria-label="Dashboard views"
                    style={{
                    display: 'flex',
                    gap: '8px',
                    flexWrap: 'wrap'
                }}>
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            id={`tab-${tab.id}`}
                            role="tab"
                            aria-selected={activeTab === tab.id}
                            tabIndex={activeTab === tab.id ? 0 : -1}
                            onKeyDown={(e) => handleTabKeyDown(e, tab.id)}
                            type="button"
                            className="ui-tab ui-focus-ring"
                            ref={(el) => {
                                if (el) tabButtonRefs.current.set(tab.id, el)
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
                    backgroundColor: '#10b981'
                }} className="ui-animate-pulse" />
                LIVE
            </div>
        </header>
    )
}
