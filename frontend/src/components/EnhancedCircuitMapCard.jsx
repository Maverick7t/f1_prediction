import React, { useState, useEffect, useRef } from 'react'
import styles from '../styles/EnhancedCircuitMapCard.module.css'

// Get API base URL from environment or default to localhost for dev
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000'

// COLORMAPS for circuit visualization
const COLORMAPS = {
    viridis: ['#440154', '#31688e', '#35b779', '#fde724'],
    plasma: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
    inferno: ['#000004', '#420a68', '#932667', '#fca238', '#fcfdbf'],
    magma: ['#000004', '#3b0f70', '#8c2981', '#fcfdbf', '#ffffff'],
    cividis: ['#00224e', '#1f4f6d', '#3d7680', '#649b8a', '#97c27d']
}

// Helper function to interpolate colors based on speed
function interpolateColor(value, cmap) {
    const colors = COLORMAPS[cmap] || COLORMAPS.viridis
    const index = Math.max(0, Math.min(value * (colors.length - 1), colors.length - 1))
    const lower = Math.floor(index)
    const upper = Math.ceil(index)

    if (lower === upper) return colors[lower]

    const t = index - lower
    const c1 = colors[lower]
    const c2 = colors[upper]

    const [r1, g1, b1] = [c1.slice(1, 3), c1.slice(3, 5), c1.slice(5, 7)].map(x => parseInt(x, 16))
    const [r2, g2, b2] = [c2.slice(1, 3), c2.slice(3, 5), c2.slice(5, 7)].map(x => parseInt(x, 16))

    const r = Math.round(r1 + (r2 - r1) * t)
    const g = Math.round(g1 + (g2 - g1) * t)
    const b = Math.round(b1 + (b2 - b1) * t)

    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// Circuit Canvas Component
function CircuitCanvas({ drivers, selectedDriver, colormap, onSelectDriver }) {
    const canvasRef = useRef(null)
    const containerRef = useRef(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || !selectedDriver) return

        const ctx = canvas.getContext('2d')
        const rect = containerRef.current.getBoundingClientRect()

        canvas.width = rect.width
        canvas.height = rect.height

        const trace = selectedDriver.circuit_trace
        if (!trace || !trace.x || trace.x.length === 0) return

        const xs = trace.x
        const ys = trace.y
        const speeds = trace.speed

        const minX = Math.min(...xs)
        const maxX = Math.max(...xs)
        const minY = Math.min(...ys)
        const maxY = Math.max(...ys)
        const minSpeed = Math.min(...speeds)
        const maxSpeed = Math.max(...speeds)

        const padding = 40
        const speedRange = maxSpeed - minSpeed || 1

        ctx.fillStyle = '#000000'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        ctx.strokeStyle = '#333333'
        ctx.lineWidth = 0.5
        for (let i = 0; i < 10; i++) {
            const x = padding + (i / 10) * (canvas.width - 2 * padding)
            const y = padding + (i / 10) * (canvas.height - 2 * padding)
            ctx.beginPath()
            ctx.moveTo(x, padding)
            ctx.lineTo(x, canvas.height - padding)
            ctx.stroke()
            ctx.beginPath()
            ctx.moveTo(padding, y)
            ctx.lineTo(canvas.width - padding, y)
            ctx.stroke()
        }

        const trackWidth = canvas.width - 2 * padding
        const trackHeight = canvas.height - 2 * padding

        for (let i = 0; i < xs.length - 1; i++) {
            const x1 = padding + ((xs[i] - minX) / (maxX - minX || 1)) * trackWidth
            const y1 = canvas.height - padding - ((ys[i] - minY) / (maxY - minY || 1)) * trackHeight
            const x2 = padding + ((xs[i + 1] - minX) / (maxX - minX || 1)) * trackWidth
            const y2 = canvas.height - padding - ((ys[i + 1] - minY) / (maxY - minY || 1)) * trackHeight

            const speedNorm = (speeds[i] - minSpeed) / speedRange
            const color = interpolateColor(speedNorm, colormap)

            ctx.strokeStyle = color
            ctx.lineWidth = 3
            ctx.lineCap = 'round'
            ctx.lineJoin = 'round'

            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()
        }

        for (let i = 0; i < xs.length; i += Math.max(1, Math.floor(xs.length / 200))) {
            const x = padding + ((xs[i] - minX) / (maxX - minX || 1)) * trackWidth
            const y = canvas.height - padding - ((ys[i] - minY) / (maxY - minY || 1)) * trackHeight

            const speedNorm = (speeds[i] - minSpeed) / speedRange
            const color = interpolateColor(speedNorm, colormap)

            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(x, y, 2, 0, Math.PI * 2)
            ctx.fill()
        }

        const startX = padding + ((xs[0] - minX) / (maxX - minX || 1)) * trackWidth
        const startY = canvas.height - padding - ((ys[0] - minY) / (maxY - minY || 1)) * trackHeight

        ctx.fillStyle = '#10b981'
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(startX, startY, 8, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()

        const endX = padding + ((xs[xs.length - 1] - minX) / (maxX - minX || 1)) * trackWidth
        const endY = canvas.height - padding - ((ys[ys.length - 1] - minY) / (maxY - minY || 1)) * trackHeight

        ctx.fillStyle = '#ef4444'
        ctx.beginPath()
        ctx.moveTo(endX - 8, endY)
        ctx.lineTo(endX + 8, endY)
        ctx.moveTo(endX, endY - 8)
        ctx.lineTo(endX, endY + 8)
        ctx.stroke()

        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 16px sans-serif'
        ctx.textAlign = 'left'
        ctx.fillText(`${selectedDriver.code} - ${selectedDriver.name}`, padding + 10, padding - 10)

        ctx.font = '12px sans-serif'
        ctx.fillStyle = '#888888'
        ctx.textAlign = 'right'
        ctx.fillText(`${minSpeed.toFixed(0)} km/h`, canvas.width - padding - 10, padding - 10)

        ctx.fillStyle = '#fde724'
        ctx.fillText(`${maxSpeed.toFixed(0)} km/h`, canvas.width - padding - 10, canvas.height - padding + 10)

    }, [selectedDriver, colormap])

    return (
        <div
            ref={containerRef}
            style={{
                position: 'relative',
                width: '100%',
                height: '500px',
                backgroundColor: '#000000',
                borderRadius: '8px',
                border: '1px solid #3a3a3a',
                overflow: 'hidden',
                cursor: 'crosshair'
            }}
        >
            <canvas
                ref={canvasRef}
                style={{
                    display: 'block',
                    width: '100%',
                    height: '100%'
                }}
            />
        </div>
    )
}

// Driver Telemetry Panel Component
function DriverTelemetryPanel({ driver }) {
    if (!driver) return null

    const stats = driver.telemetry_stats

    return (
        <div className={styles['telemetry-panel']}>
            <div className={styles['panel-header']}>
                <div className={styles['driver-badge']}>
                    <div className={styles['driver-code']}>{driver.code}</div>
                </div>
                <div className={styles['driver-info']}>
                    <div className={styles['driver-name']}>{driver.name}</div>
                    <div className={styles['driver-team']}>{driver.team}</div>
                    <div className={styles['driver-position']}>Qualified: P{driver.qualifying_position}</div>
                </div>
            </div>

            <div className={styles['tab-content']}>
                <div className={styles['tab-pane']}>
                    <div className={styles['stat-group']}>
                        <div className={styles['stat-label']}>LAP TIME</div>
                        <div className={styles['stat-value']}>{stats.lap_time_s.toFixed(3)}s</div>
                    </div>

                    <div className={styles['stat-divider']}></div>

                    <div className={styles['stat-group']}>
                        <div className={styles['stat-label']}>SECTORS</div>
                        <div className={styles['sector-grid']}>
                            <div className={styles['sector-item']}>
                                <div className={styles['sector-label']}>S1</div>
                                <div className={styles['sector-time']}>{stats.sector1_s.toFixed(2)}s</div>
                            </div>
                            <div className={styles['sector-item']}>
                                <div className={styles['sector-label']}>S2</div>
                                <div className={styles['sector-time']}>{stats.sector2_s.toFixed(2)}s</div>
                            </div>
                            <div className={styles['sector-item']}>
                                <div className={styles['sector-label']}>S3</div>
                                <div className={styles['sector-time']}>{stats.sector3_s.toFixed(2)}s</div>
                            </div>
                        </div>
                    </div>

                    <div className={styles['stat-divider']}></div>

                    <div className={styles['stat-group']}>
                        <div className={styles['stat-label']}>SPEED ANALYSIS</div>
                        <div className={styles['speed-stats']}>
                            <div className={styles['speed-row']}>
                                <span className={styles['speed-label']}>Top Speed</span>
                                <span className={styles['speed-value']}>{stats.top_speed_kmh}km/h</span>
                            </div>
                            <div className={styles['speed-row']}>
                                <span className={styles['speed-label']}>Avg Speed</span>
                                <span className={styles['speed-value']}>{stats.avg_speed_kmh}km/h</span>
                            </div>
                            <div className={styles['speed-row']}>
                                <span className={styles['speed-label']}>Min Speed</span>
                                <span className={styles['speed-value']}>{stats.min_speed_kmh}km/h</span>
                            </div>
                        </div>
                    </div>

                    <div className={styles['stat-divider']}></div>

                    <div className={styles['stat-group']}>
                        <div className={styles['stat-label']}>DATA POINTS</div>
                        <div className={styles['stat-value']} style={{ fontSize: '14px' }}>
                            {stats.total_data_points.toLocaleString()} samples
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

// Driver Selector Component
function DriverSelector({ drivers, selectedDriver, onSelectDriver }) {
    const getTeamColor = (teamName) => {
        const colors = {
            'McLaren': '#ea580c',
            'Mercedes': '#64748b',
            'Red Bull': '#0052cc',
            'Red Bull Racing': '#0052cc',
            'Ferrari': '#dc2626',
            'Haas': '#e5e5e5',
            'Aston Martin': '#006241',
            'Alpine': '#0082fa',
            'Racing Bulls': '#1e3a8a',
            'Kick Sauber': '#c8102e',
            'Williams': '#0082fa'
        }
        return colors[teamName] || '#94a3b8'
    }

    return (
        <div className={styles['driver-selector-container']}>
            <div className={styles['selector-label']}>Top 6 Drivers</div>
            <div className={styles['driver-buttons']}>
                {drivers.map((driver) => (
                    <button
                        key={driver.code}
                        className={`${styles['driver-button']} ${selectedDriver?.code === driver.code ? styles['active'] : ''}`}
                        onClick={() => onSelectDriver(driver)}
                        style={{
                            borderColor: selectedDriver?.code === driver.code ? getTeamColor(driver.team) : '#3a3a3a',
                            backgroundColor: selectedDriver?.code === driver.code
                                ? `${getTeamColor(driver.team)}20`
                                : 'rgba(80, 80, 80, 0.3)'
                        }}
                    >
                        <div
                            className={styles['driver-avatar']}
                            style={{ backgroundColor: getTeamColor(driver.team) }}
                        >
                            {driver.code.charAt(0)}
                        </div>
                        <div className={styles['driver-btn-info']}>
                            <div className={styles['driver-btn-code']}>{driver.code}</div>
                            <div className={styles['driver-btn-name']}>{driver.name.split(' ')[1]}</div>
                            <div className={styles['driver-btn-detail']}>P{driver.qualifying_position}</div>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    )
}

// Main Component
export default function EnhancedCircuitMapCard() {
    const [drivers, setDrivers] = useState([])
    const [selectedDriver, setSelectedDriver] = useState(null)
    const [sessionInfo, setSessionInfo] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [colormap, setColormap] = useState('viridis')

    useEffect(() => {
        async function loadQualifyingTelemetry() {
            try {
                setLoading(true)
                setError(null)

                console.log('📊 Loading qualifying circuit telemetry...')

                const response = await fetch(`${API_BASE}/api/qualifying-circuit-telemetry`)
                const data = await response.json()

                if (!data.success) {
                    throw new Error(data.error || 'Failed to load telemetry')
                }

                console.log('✓ Telemetry loaded:', data)
                setSessionInfo(data.session_info)
                setDrivers(data.drivers)

                if (data.drivers.length > 0) {
                    setSelectedDriver(data.drivers[0])
                }
            } catch (err) {
                console.error('❌ Error loading telemetry:', err)
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        loadQualifyingTelemetry()
    }, [])

    const handleDriverSelect = (driver) => {
        setSelectedDriver(driver)
    }

    const handleColormapChange = (e) => {
        setColormap(e.target.value)
    }

    return (
        <div className={styles['enhanced-circuit-container']}>
            <div className={styles['circuit-header']}>
                <div className={styles['header-title']}>
                    <span className={styles['icon']}>🏁</span>
                    CIRCUIT ANALYSIS - QUALIFYING TELEMETRY
                </div>
                {sessionInfo && (
                    <div className={styles['header-info']}>
                        Round {sessionInfo.round} • {sessionInfo.event} • {sessionInfo.year}
                    </div>
                )}
            </div>

            {error && (
                <div className={styles['error-banner']}>
                    <span className={styles['error-icon']}>⚠️</span>
                    {error}
                </div>
            )}

            {!loading && drivers.length > 0 && selectedDriver ? (
                <div className={styles['circuit-content']}>
                    <div className={styles['circuit-canvas-wrapper']}>
                        <CircuitCanvas
                            drivers={drivers}
                            selectedDriver={selectedDriver}
                            colormap={colormap}
                            onSelectDriver={handleDriverSelect}
                        />
                    </div>

                    <div className={styles['telemetry-panel-wrapper']}>
                        <DriverTelemetryPanel driver={selectedDriver} />
                    </div>
                </div>
            ) : (
                <div className={styles['loading-container']}>
                    <div className={styles['spinner']}></div>
                    <p>{loading ? 'Loading qualifying data...' : 'No telemetry data available'}</p>
                </div>
            )}

            {drivers.length > 0 && (
                <div className={styles['circuit-controls']}>
                    <div className={styles['colormap-selector']}>
                        <label>Colormap:</label>
                        <select value={colormap} onChange={handleColormapChange}>
                            <option value="viridis">Viridis</option>
                            <option value="plasma">Plasma</option>
                            <option value="inferno">Inferno</option>
                            <option value="magma">Magma</option>
                            <option value="cividis">Cividis</option>
                        </select>
                    </div>
                </div>
            )}

            {drivers.length > 0 && (
                <DriverSelector
                    drivers={drivers}
                    selectedDriver={selectedDriver}
                    onSelectDriver={handleDriverSelect}
                />
            )}
        </div>
    )
}
