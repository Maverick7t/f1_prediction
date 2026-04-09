import React, { useEffect, useMemo, useState } from 'react'
import Header from './Header'
import DriverCard from './DriverCard'
import RaceInfoCard from './RaceInfoCard'
import WinnerPredictionCard from './WinnerPredictionCard'
import RaceHistoryCard from './RaceHistoryCard'
import SeasonReviewCard from './SeasonReviewCard'
import MatchupCard from './MatchupCard'
import EnhancedCircuitMapCard from './EnhancedCircuitMapCard'
import StandingsView from './StandingsView'
import ConstructorStandingsView from './ConstructorStandingsView'
import ModelMetricsCard from './ModelMetricsCard'
import { fetchSaoPauloPredictions, transformPredictionsToDriverData, fetchNextRace, fetchCurrentDrivers, fetchConstructorStandings, fetchRaceHistory, fetchSeasonReview } from '../api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function coerceValidDate(value) {
  if (!value) return null
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value
  }
  if (typeof value === 'string' || typeof value === 'number') {
    const parsed = new Date(value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  }
  return null
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('current')
  const [driverData, setDriverData] = useState([])
  const [winnerPrediction, setWinnerPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [nextRace, setNextRace] = useState(null)
  const [openF1Drivers, setOpenF1Drivers] = useState([])

  const openF1DriverByAcronym = useMemo(() => {
    const map = new Map()
    for (const driver of openF1Drivers || []) {
      const acronym = (driver?.nameAcronym || '').toUpperCase()
      if (acronym) map.set(acronym, driver)
    }
    return map
  }, [openF1Drivers])

  const [constructorStandings, setConstructorStandings] = useState([])
  const [raceHistory, setRaceHistory] = useState([])
  const [seasonReview, setSeasonReview] = useState(null)
  const [isOffSeason, setIsOffSeason] = useState(false)

  // Fetch predictions on component mount
  useEffect(() => {
    let cancelled = false

    const DASHBOARD_CACHE_KEY = 'f1_dashboard_cache_v1'
    const DASHBOARD_CACHE_TTL_MS = 12 * 60 * 60 * 1000 // 12 hours

    function reviveCachedDashboard(parsed) {
      if (!parsed || typeof parsed !== 'object') return parsed
      if (!parsed.nextRace || typeof parsed.nextRace !== 'object') return parsed

      return {
        ...parsed,
        nextRace: {
          ...parsed.nextRace,
          dateStart: coerceValidDate(parsed.nextRace.dateStart),
          dateEnd: coerceValidDate(parsed.nextRace.dateEnd)
        }
      }
    }

    function loadDashboardCache() {
      try {
        const raw = localStorage.getItem(DASHBOARD_CACHE_KEY)
        if (!raw) return null
        const parsed = reviveCachedDashboard(JSON.parse(raw))
        const savedAt = parsed?.savedAt
        if (!savedAt || typeof savedAt !== 'number') return null
        if (Date.now() - savedAt > DASHBOARD_CACHE_TTL_MS) return null
        return parsed
      } catch {
        return null
      }
    }

    function saveDashboardCache(payload) {
      try {
        localStorage.setItem(DASHBOARD_CACHE_KEY, JSON.stringify({
          savedAt: Date.now(),
          ...payload
        }))
      } catch {
        // ignore quota/security errors
      }
    }

    const cached = loadDashboardCache()
    const hasCachedData = Boolean(cached)

    // Hydrate UI immediately from last-known data to mask backend cold starts.
    if (cached && !cancelled) {
      if (cached.nextRace !== undefined) setNextRace(cached.nextRace)
      if (cached.openF1Drivers !== undefined) setOpenF1Drivers(cached.openF1Drivers || [])
      if (cached.constructorStandings !== undefined) setConstructorStandings(cached.constructorStandings || [])
      if (cached.raceHistory !== undefined) setRaceHistory(cached.raceHistory || [])

      const cachedPredictions = cached.predictions
      if (cachedPredictions) {
        if (cachedPredictions.isOffSeason && cachedPredictions.seasonReview) {
          setIsOffSeason(true)
          setSeasonReview(cachedPredictions.seasonReview)
        } else {
          setIsOffSeason(false)
        }

        const drivers = transformPredictionsToDriverData(cachedPredictions)
        setDriverData(drivers)

        if (cachedPredictions.winner_prediction) {
          setWinnerPrediction({
            driver: cachedPredictions.winner_prediction.driver,
            team: cachedPredictions.winner_prediction.team,
            percentage: cachedPredictions.winner_prediction.percentage
          })
        }

        setError(null)
      }

      setLoading(false)
    }

    async function loadPredictions() {
      try {
        // Only show the blocking loader when we don't have any cached UI state.
        if (!hasCachedData) setLoading(true)

        // Run all independent API calls in parallel with timeouts
        const [raceResult, driversResult, constructorsResult, raceHistoryResult, predictionsResult] =
          await Promise.allSettled([
            fetchNextRace().catch(err => ({ error: err.message })),
            fetchCurrentDrivers().catch(() => []),
            fetchConstructorStandings().catch(() => []),
            fetchRaceHistory().catch(() => []),
            fetchSaoPauloPredictions()
          ])

        // If this effect was cleaned up (StrictMode), bail out
        if (cancelled) return

        // Process next race
        if (raceResult.status === 'fulfilled') {
          setNextRace(raceResult.value)
          console.log('✓ Next race loaded:', raceResult.value?.raceName)
        } else {
          console.error('❌ Failed to load next race info:', raceResult.reason)
          if (!hasCachedData) setNextRace({ error: raceResult.reason?.message })
        }

        // Process drivers
        if (driversResult.status === 'fulfilled') {
          setOpenF1Drivers(driversResult.value || [])
          console.log(`✓ Loaded ${(driversResult.value || []).length} drivers from OpenF1`)
        }

        // Process constructor standings
        if (constructorsResult.status === 'fulfilled') {
          setConstructorStandings(constructorsResult.value || [])
          console.log('✓ Constructor standings loaded')
        }

        // Process race history
        if (raceHistoryResult.status === 'fulfilled') {
          setRaceHistory(raceHistoryResult.value || [])
          console.log('✓ Race history loaded:', (raceHistoryResult.value || []).length, 'races')
        } else {
          if (!hasCachedData) setRaceHistory([])
        }

        // Process predictions
        if (predictionsResult.status === 'fulfilled') {
          const predictions = predictionsResult.value

          // Check if we're in off-season mode
          if (predictions.isOffSeason && predictions.seasonReview) {
            setIsOffSeason(true)
            setSeasonReview(predictions.seasonReview)
            console.log(`✓ Off-season mode: showing ${predictions.seasonReview.year} season review`)
          } else {
            setIsOffSeason(false)
          }

          // Transform predictions to driver data
          const drivers = transformPredictionsToDriverData(predictions)
          setDriverData(drivers)

          // Set winner prediction
          if (predictions.winner_prediction) {
            setWinnerPrediction({
              driver: predictions.winner_prediction.driver,
              team: predictions.winner_prediction.team,
              percentage: predictions.winner_prediction.percentage
            })
          }

          setError(null)
          console.log('✓ All predictions loaded from backend')

          // Persist last successful payloads for cold-start masking
          saveDashboardCache({
            nextRace: raceResult.status === 'fulfilled' ? raceResult.value : cached?.nextRace,
            openF1Drivers: driversResult.status === 'fulfilled' ? (driversResult.value || []) : (cached?.openF1Drivers || []),
            constructorStandings: constructorsResult.status === 'fulfilled' ? (constructorsResult.value || []) : (cached?.constructorStandings || []),
            raceHistory: raceHistoryResult.status === 'fulfilled' ? (raceHistoryResult.value || []) : (cached?.raceHistory || []),
            predictions
          })
        } else {
          console.error('❌ Failed to load predictions:', predictionsResult.reason)

          // If we already hydrated from cache, keep showing stale data.
          if (!hasCachedData) {
            // Try to load season review directly as fallback
            try {
              if (cancelled) return
              const review = await fetchSeasonReview()
              if (cancelled) return
              setSeasonReview(review)
              setIsOffSeason(true)
              console.log(`✓ Fallback: loaded ${review.year} season review`)
            } catch (reviewErr) {
              console.error('❌ Season review fallback also failed:', reviewErr)
            }

            setError('Failed to load predictions from API. Showing season review instead.')
            setDriverData([])
            setWinnerPrediction(null)
          }
        }
      } catch (outerErr) {
        console.error('❌ Unexpected error in loadPredictions:', outerErr)
        if (!cancelled) {
          setError('An unexpected error occurred while loading data.')
        }
      } finally {
        if (!cancelled && !hasCachedData) {
          setLoading(false)
        }
      }
    }

    loadPredictions()

    return () => {
      cancelled = true
    }
  }, [])

  const nextRaceDateStart = coerceValidDate(nextRace?.dateStart)

  return (
    <div className="dashboard-root" aria-busy={loading} style={{
      backgroundColor: '#1a1a1a',
      minHeight: '100vh',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      color: '#e2e8f0'
    }}>
      {/* Header Component */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Loading State */}
      {loading && (
        <div style={{
          textAlign: 'center',
          padding: '60px',
          color: '#94a3b8',
          fontSize: '14px'
        }}>
          <div style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px'
          }}>Loading Predictions...</div>
          <div style={{ fontSize: '14px' }}>Fetching data from ML model...</div>
        </div>
      )}

      {/* Error State - only show if NOT in off-season mode */}
      {error && !loading && !isOffSeason && (
        <div style={{
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '10px',
          padding: '20px',
          margin: '20px 0',
          color: '#fca5a5',
          fontSize: '14px'
        }}>
          <div style={{ fontWeight: '700', marginBottom: '8px' }}>⚠️ {error}</div>
          <div style={{ fontSize: '13px', opacity: 0.8 }}>Showing fallback data. Make sure the API server is running on {API_BASE_URL}</div>
        </div>
      )}

      {/* Current Race Tab Content */}
      {!loading && activeTab === 'current' && (
        isOffSeason && seasonReview ? (
          /* OFF-SEASON: Show Season Review */
          <div
            className="dashboard-wide"
            role="tabpanel"
            id="panel-current"
            aria-labelledby="tab-current"
            tabIndex={0}
            style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '24px'
          }}>
            {/* Off-Season Banner */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              padding: '20px 28px',
              display: 'flex',
              alignItems: 'center',
              gap: '16px'
            }}>
              <div style={{
                fontSize: '28px',
                lineHeight: 1
              }}>
                🏁
              </div>
              <div>
                <div style={{
                  fontSize: '16px',
                  fontWeight: '700',
                  color: '#f59e0b',
                  marginBottom: '4px'
                }}>
                  Season Not Yet Started
                </div>
                <div style={{
                  fontSize: '14px',
                  color: '#94a3b8'
                }}>
                  The {new Date().getFullYear()} season hasn't begun yet. Here's how our model performed during the {seasonReview.year} season.
                </div>
              </div>
            </div>

            {/* Season Review Card */}
            <SeasonReviewCard
              seasonReview={seasonReview}
              onYearChange={async (year) => {
                try {
                  const review = await fetchSeasonReview(year)
                  setSeasonReview(review)
                } catch (err) {
                  console.error('Failed to load season review for year:', year, err)
                }
              }}
            />
          </div>
        ) : driverData.length === 0 && error && !seasonReview ? (
          <div
            role="tabpanel"
            id="panel-current"
            aria-labelledby="tab-current"
            tabIndex={0}
            style={{
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '10px',
            padding: '40px',
            textAlign: 'center',
            maxWidth: '600px'
          }}>
            <div style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#fca5a5',
              marginBottom: '12px'
            }}>
              Unable to Load Data
            </div>
            <div style={{
              fontSize: '14px',
              color: '#cbd5e1',
              marginBottom: '20px'
            }}>
              {error}
            </div>
            <div style={{
              fontSize: '13px',
              color: '#94a3b8'
            }}>
              Please check that:
              <ul style={{
                listStyle: 'none',
                padding: 0,
                marginTop: '12px',
                textAlign: 'left',
                maxWidth: '400px',
                margin: '12px auto 0'
              }}>
                <li>✓ Backend server is running on {API_BASE_URL}</li>
                <li>✓ VITE_API_URL environment variable is set correctly</li>
                <li>✓ OpenF1 API is accessible</li>
              </ul>
            </div>
          </div>
        ) : (
          <div
            className="dashboard-grid-main"
            role="tabpanel"
            id="panel-current"
            aria-labelledby="tab-current"
            tabIndex={0}
          >
            {/* Left Column */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '20px'
            }}>
              <RaceInfoCard
                raceName={nextRace?.error ? 'Next race unavailable' : (nextRace?.raceName || 'TBD')}
                dates={nextRaceDateStart ? nextRaceDateStart.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }).toUpperCase() : 'TBD'}
                time={nextRaceDateStart ? nextRaceDateStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }) : 'TBD'}
                track={nextRace?.error ? 'TBD' : (nextRace?.circuitName || 'TBD')}
                country={nextRace?.error ? undefined : nextRace?.country}
                circuitImage={nextRace?.error ? null : nextRace?.circuitImage}
              />
              <WinnerPredictionCard
                percentage={winnerPrediction?.percentage || 72}
                driverName={winnerPrediction?.driver || 'NOR'}
                teamColor="#ea580c"
                headshotUrl={openF1DriverByAcronym.get((winnerPrediction?.driver || '').toUpperCase())?.headshotUrl}
                fullName={openF1DriverByAcronym.get((winnerPrediction?.driver || '').toUpperCase())?.fullName}
                confidence={winnerPrediction?.confidence || 'HIGH'}
                confidenceColor={winnerPrediction?.confidence_color || '#f59e0b'}
              />

            </div>

            {/* Right Column */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '20px'
            }}>
              {/* Driver Predictions Grid - 2x3 */}
              <div className="driver-grid">
                {driverData.map((driver, idx) => {
                  const openF1Driver = openF1DriverByAcronym.get((driver?.name || '').toUpperCase())

                  return (
                    <DriverCard
                      key={`${driver?.name || 'driver'}-${driver?.team || 'team'}-${driver?.position || idx}`}
                      name={driver.name}
                      team={driver.team}
                      percentage={driver.percentage}
                      teamColor={openF1Driver?.teamColour || driver.teamColor}
                      position={driver.position}
                      points={driver.points}
                      headshotUrl={openF1Driver?.headshotUrl}
                      fullName={openF1Driver?.fullName}
                      confidence={driver.confidence || 'MEDIUM'}
                      confidenceColor={driver.confidenceColor || '#f59e0b'}
                    />
                  );
                })}
              </div>

              {/* Race History Table */}
              <RaceHistoryCard raceHistory={raceHistory} />
            </div>
          </div>
        )
      )}

      {/* Driver Standings Tab */}
      {!loading && activeTab === 'standings' && (
        <div
          className="dashboard-single"
          role="tabpanel"
          id="panel-standings"
          aria-labelledby="tab-standings"
          tabIndex={0}
        >
          <StandingsView />
        </div>
      )}

      {/* Constructor Standings Tab */}
      {!loading && activeTab === 'constructor' && (
        <div
          className="dashboard-single"
          role="tabpanel"
          id="panel-constructor"
          aria-labelledby="tab-constructor"
          tabIndex={0}
        >
          <ConstructorStandingsView />
        </div>
      )}

      {/* Circuit Map Tab */}
      {!loading && activeTab === 'circuit' && (
        <div
          className="dashboard-wide"
          role="tabpanel"
          id="panel-circuit"
          aria-labelledby="tab-circuit"
          tabIndex={0}
          style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '28px'
        }}>
          <EnhancedCircuitMapCard />
        </div>
      )}

      {/* Matchup Tab */}
      {!loading && activeTab === 'matchup' && (
        <div
          className="dashboard-grid-matchup"
          role="tabpanel"
          id="panel-matchup"
          aria-labelledby="tab-matchup"
          tabIndex={0}
        >
          <MatchupCard drivers={driverData} />
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '20px'
          }}>
            <div style={{
              backgroundColor: '#1a1f2e',
              border: '1px solid #334155',
              borderRadius: '10px',
              padding: '18px'
            }}>
              <div style={{
                fontSize: '13px',
                fontWeight: '700',
                letterSpacing: '0.5px',
                color: '#94a3b8',
                textTransform: 'uppercase',
                marginBottom: '12px'
              }}>
                HOW THIS WORKS
              </div>
              <div style={{
                fontSize: '14px',
                color: '#cbd5e1',
                lineHeight: '1.8'
              }}>
                <p>Select any two drivers to compare their predicted performance in the upcoming race.</p>
                <p style={{ marginTop: '12px' }}>The model analyzes:</p>
                <ul style={{ marginLeft: '16px', marginTop: '8px', listStyle: 'none' }}>
                  <li>📊 Recent form & race results</li>
                  <li>🏁 Circuit-specific performance</li>
                  <li>⛅ Weather predictions</li>
                  <li>🔧 Vehicle setup & reliability</li>
                  <li>👥 Head-to-head history</li>
                </ul>
              </div>
            </div>
            <div style={{
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '10px',
              padding: '16px'
            }}>
              <div style={{
                fontSize: '13px',
                fontWeight: '700',
                color: '#10b981',
                marginBottom: '8px'
              }}>
                💡 PREDICTION TIP
              </div>
              <div style={{
                fontSize: '13px',
                color: '#cbd5e1',
                lineHeight: '1.6'
              }}>
                Check for weather updates 24 hours before the race - wet conditions can dramatically shift these probabilities!
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MLOps Model Monitor Tab */}
      {!loading && activeTab === 'mlops' && (
        <div
          className="dashboard-single"
          role="tabpanel"
          id="panel-mlops"
          aria-labelledby="tab-mlops"
          tabIndex={0}
          style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '28px'
        }}>
          <ModelMetricsCard />
        </div>
      )}

    </div>
  )
}
