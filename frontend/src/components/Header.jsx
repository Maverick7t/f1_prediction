import React from 'react'

export default function Header({ activeTab, setActiveTab }) {
    const [logoFailed, setLogoFailed] = React.useState(false)

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
        <header className="sticky top-0 z-50 mb-6">
            <div className="mx-auto w-full max-w-[1440px] rounded-md border border-(--color-border) bg-[rgba(0,0,0,0.4)] px-4 py-3 backdrop-blur">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    {/* Brand */}
                    <button
                        type="button"
                        onClick={() => setActiveTab('current')}
                        className="ui-focus-ring flex items-center gap-2 rounded-md px-2 py-1"
                        aria-label="Go to current race dashboard"
                    >
                        <img
                            src="/logo.png"
                            alt="F1"
                            className="h-6 w-auto"
                            onError={() => setLogoFailed(true)}
                        />
                        <div className="text-lg font-extrabold tracking-[0.25em] text-white">
                            PREDICT
                        </div>
                    </button>

                    {/* Navigation */}
                    <nav
                        role="tablist"
                        aria-label="Dashboard views"
                        aria-orientation="horizontal"
                        className="flex flex-wrap gap-2"
                    >
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                id={`tab-${tab.id}`}
                                role="tab"
                                aria-selected={activeTab === tab.id}
                                aria-controls={`panel-${tab.id}`}
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
            </div>
        </header>
    )
}
