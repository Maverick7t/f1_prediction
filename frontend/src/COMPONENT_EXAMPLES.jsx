/**
 * F1 PREDICT - Component Usage Examples & Patterns
 * 
 * This file demonstrates how to use each component with different configurations
 * and how to integrate with backend APIs
 */

import React, { useState, useEffect, createContext, useContext } from 'react'
import { mockDrivers, mockRaces, mockLiveUpdates } from '../mockData'

// ============ EXAMPLE 1: Using Mock Data =============
export const DashboardWithMockData = () => {

    return (
        <Dashboard>
            {mockDrivers.map(driver => (
                <DriverCard key={driver.id} {...driver} />
            ))}
        </Dashboard>
    )
}

// ============ EXAMPLE 2: Data Fetching from Backend =============
export const DashboardWithAPI = () => {
    const [drivers, setDrivers] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        // Replace with your actual API endpoint
        fetch('/api/predictions/drivers')
            .then(res => res.json())
            .then(data => {
                setDrivers(data)
                setLoading(false)
            })
            .catch(err => {
                setError(err.message)
                setLoading(false)
            })
    }, [])

    if (loading) return <div>Loading predictions...</div>
    if (error) return <div>Error: {error}</div>

    return (
        <div>
            {drivers.map(driver => (
                <DriverCard key={driver.id} {...driver} />
            ))}
        </div>
    )
}

// ============ EXAMPLE 3: Real-time Updates with WebSocket =============
export const DashboardWithRealtimeUpdates = () => {
    const [updates, setUpdates] = useState([])

    useEffect(() => {
        const ws = new WebSocket('ws://your-api.com/live-updates')

        ws.onmessage = (event) => {
            const newUpdate = JSON.parse(event.data)
            setUpdates(prev => [newUpdate, ...prev.slice(0, 4)])
        }

        return () => ws.close()
    }, [])

    return <LiveUpdatesCard updates={updates} />
}

// ============ EXAMPLE 4: Dynamic Data with State Management =============
export const DashboardWithStateManagement = () => {
    const [selectedDriver, setSelectedDriver] = useState(null)
    const [predictions, setPredictions] = useState({})
    const [isLoading, setIsLoading] = useState(false)

    const handleDriverSelect = async (driverId) => {
        setSelectedDriver(driverId)
        setIsLoading(true)

        try {
            const response = await fetch(`/api/driver/${driverId}/predictions`)
            const data = await response.json()
            setPredictions(data)
        } catch (error) {
            console.error('Error fetching predictions:', error)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div>
            {isLoading && <LoadingSpinner />}
            {selectedDriver && (
                <DriverCard {...predictions} />
            )}
        </div>
    )
}

// ============ EXAMPLE 5: Context API for Global State =============
// Create context
const PredictionContext = createContext()

export const PredictionProvider = ({ children }) => {
    const [activeRace, setActiveRace] = useState(null)
    const [drivers, setDrivers] = useState([])
    const [predictions, setPredictions] = useState({})

    useEffect(() => {
        // Fetch all data on mount
        fetchRaceData()
    }, [activeRace])

    const fetchRaceData = async () => {
        // Fetch from API
    }

    return (
        <PredictionContext.Provider
            value={{ activeRace, setActiveRace, drivers, predictions }}
        >
            {children}
        </PredictionContext.Provider>
    )
}

export const usePredictions = () => {
    const context = useContext(PredictionContext)
    if (!context) {
        throw new Error('usePredictions must be used within PredictionProvider')
    }
    return context
}

// Usage in component
export const MyComponent = () => {
    const { drivers, predictions } = usePredictions()

    return (
        <div>
            {drivers.map(driver => (
                <DriverCard key={driver.id} {...driver} />
            ))}
        </div>
    )
}

// ============ EXAMPLE 6: Custom Hooks for Data Fetching =============
export const useFetchDrivers = (raceId) => {
    const [drivers, setDrivers] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true)
                const response = await fetch(`/api/races/${raceId}/drivers`)
                if (!response.ok) throw new Error('Failed to fetch')
                const data = await response.json()
                setDrivers(data)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        fetchData()
    }, [raceId])

    return { drivers, loading, error }
}

// Usage
export const RaceDriversDisplay = ({ raceId }) => {
    const { drivers, loading, error } = useFetchDrivers(raceId)

    if (loading) return <LoadingSpinner />
    if (error) return <ErrorDisplay message={error} />

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '14px' }}>
            {drivers.map(driver => (
                <DriverCard key={driver.id} {...driver} />
            ))}
        </div>
    )
}

// ============ EXAMPLE 7: Error Boundary Component =============
export class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props)
        this.state = { hasError: false, error: null }
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error }
    }

    componentDidCatch(error, errorInfo) {
        console.error('Error caught by boundary:', error, errorInfo)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    backgroundColor: '#1a1f2e',
                    border: '1px solid #dc2626',
                    borderRadius: '10px',
                    padding: '16px',
                    color: '#ef4444'
                }}>
                    <h2>Something went wrong</h2>
                    <p>{this.state.error?.message}</p>
                </div>
            )
        }

        return this.props.children
    }
}

// Usage
export const App = () => (
    <ErrorBoundary>
        <Dashboard />
    </ErrorBoundary>
)

// ============ EXAMPLE 8: Styled Wrapper Components =============
export const DashboardContainer = ({ children }) => (
    <div style={{
        backgroundColor: '#0b0f14',
        minHeight: '100vh',
        padding: '24px 32px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        color: '#e2e8f0'
    }}>
        {children}
    </div>
)

export const Card = ({ children, hoverable = true }) => (
    <div style={{
        backgroundColor: '#1a1f2e',
        border: '1px solid #334155',
        borderRadius: '10px',
        padding: '16px',
        transition: hoverable ? 'all 0.3s ease' : 'none',
        cursor: hoverable ? 'pointer' : 'default',
        ':hover': hoverable ? {
            backgroundColor: '#252d3d',
            borderColor: '#06b6d4'
        } : {}
    }}>
        {children}
    </div>
)

// ============ EXAMPLE 9: Responsive Layout Pattern =============
export const ResponsiveGrid = ({ children, columns = 2 }) => (
    <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: '14px',
        '@media (max-width: 1024px)': {
            gridTemplateColumns: columns > 1 ? '1fr' : columns,
        },
        '@media (max-width: 768px)': {
            gridTemplateColumns: '1fr',
        }
    }}>
        {children}
    </div>
)

// ============ EXAMPLE 10: Theme Switcher =============
export const useTheme = () => {
    const [theme, setTheme] = useState('dark')

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark')
    }

    const colors = theme === 'dark' ? {
        bg: '#0b0f14',
        card: '#1a1f2e',
        text: '#e2e8f0'
    } : {
        bg: '#f8f9fa',
        card: '#ffffff',
        text: '#1a1f2e'
    }

    return { theme, toggleTheme, colors }
}

// ============ EXAMPLE 11: Pagination for Large Lists =============
export const usePagination = (items, itemsPerPage = 10) => {
    const [currentPage, setCurrentPage] = useState(1)

    const totalPages = Math.ceil(items.length / itemsPerPage)
    const startIndex = (currentPage - 1) * itemsPerPage
    const paginatedItems = items.slice(startIndex, startIndex + itemsPerPage)

    return {
        paginatedItems,
        currentPage,
        totalPages,
        goToPage: setCurrentPage,
        nextPage: () => setCurrentPage(prev => Math.min(prev + 1, totalPages)),
        prevPage: () => setCurrentPage(prev => Math.max(prev - 1, 1))
    }
}

// ============ EXAMPLE 12: Memoized Components for Performance =============
export const MemoizedDriverCard = React.memo(DriverCard, (prevProps, nextProps) => {
    return (
        prevProps.id === nextProps.id &&
        prevProps.percentage === nextProps.percentage &&
        prevProps.position === nextProps.position
    )
})

// ============ EXAMPLE 13: Combining Multiple Features =============
export const AdvancedDashboard = () => {
    const [activeTab, setActiveTab] = useState('current')
    const [selectedDriver, setSelectedDriver] = useState(null)
    const { raceId } = useRoute() // Assuming you have routing
    const { drivers, loading, error } = useFetchDrivers(raceId)
    const { colors } = useTheme()
    const { paginatedItems } = usePagination(drivers, 6)

    if (loading) return <LoadingSpinner />
    if (error) return <ErrorDisplay message={error} />

    return (
        <DashboardContainer>
            <Header activeTab={activeTab} setActiveTab={setActiveTab} />

            {activeTab === 'current' && (
                <ResponsiveGrid columns={2}>
                    {paginatedItems.map(driver => (
                        <MemoizedDriverCard
                            key={driver.id}
                            {...driver}
                            isSelected={selectedDriver?.id === driver.id}
                            onClick={() => setSelectedDriver(driver)}
                        />
                    ))}
                </ResponsiveGrid>
            )}
        </DashboardContainer>
    )
}

// ============ EXAMPLE 14: Form Integration for Matchup Feature =============
export const MatchupForm = () => {
    const [formData, setFormData] = useState({
        driver1: '',
        driver2: '',
        raceConditions: 'dry'
    })

    const handleSubmit = async (e) => {
        e.preventDefault()

        const response = await fetch('/api/predictions/matchup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        })

        const predictions = await response.json()
        // Handle predictions...
    }

    return (
        <form onSubmit={handleSubmit}>
            <select
                value={formData.driver1}
                onChange={(e) => setFormData({ ...formData, driver1: e.target.value })}
            >
                {/* Driver options */}
            </select>
            <select
                value={formData.driver2}
                onChange={(e) => setFormData({ ...formData, driver2: e.target.value })}
            >
                {/* Driver options */}
            </select>
            <select
                value={formData.raceConditions}
                onChange={(e) => setFormData({ ...formData, raceConditions: e.target.value })}
            >
                <option value="dry">Dry</option>
                <option value="wet">Wet</option>
                <option value="mixed">Mixed</option>
            </select>
            <button type="submit">Get Prediction</button>
        </form>
    )
}

export default {
    DashboardWithMockData,
    DashboardWithAPI,
    DashboardWithRealtimeUpdates,
    DashboardWithStateManagement,
    PredictionProvider,
    useFetchDrivers,
    ErrorBoundary,
    AdvancedDashboard,
    MatchupForm,
}
