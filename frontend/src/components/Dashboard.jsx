import React, { useEffect, useMemo, useState } from 'react'
import Header from './Header'
import DriverCard from './DriverCard'
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

function formatCountdown(targetDate) {
  if (!targetDate) return 'TBD'
  const diffMs = targetDate.getTime() - Date.now()
  if (!Number.isFinite(diffMs)) return 'TBD'
  if (diffMs <= 0) return 'IN PROGRESS / COMPLETE'

  const totalMinutes = Math.floor(diffMs / (60 * 1000))
  const days = Math.floor(totalMinutes / (60 * 24))
  const hours = Math.floor((totalMinutes - days * 60 * 24) / 60)
  const minutes = totalMinutes - days * 60 * 24 - hours * 60

  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function computeLast5ModelHealth(raceHistory) {
  const races = Array.isArray(raceHistory) ? raceHistory : []
  const byDateDesc = [...races].sort((a, b) => new Date(b?.date) - new Date(a?.date))
  const last5 = byDateDesc.slice(0, 5)
  const total = last5.length
  const correct = last5.filter(r => Boolean(r?.correct)).length
  const accuracy = total ? Math.round((correct / total) * 100) : null

  const confidences = last5
    .map(r => (typeof r?.confidence === 'number' ? r.confidence : null))
    .filter(v => typeof v === 'number')

  const avgConfidence = confidences.length
    ? Math.round(confidences.reduce((sum, v) => sum + v, 0) / confidences.length)
    : null

  return { last5, total, correct, accuracy, avgConfidence }
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
  const modelHealth = useMemo(() => computeLast5ModelHealth(raceHistory), [raceHistory])

  return (
    <div className="dashboard-root" aria-busy={loading}>
      {/* Header Component */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Loading State */}
      {loading && (
        <div className="dashboard-wide rounded-lg border border-(--color-border) bg-(--color-surface) p-10 text-center">
          <div className="ui-animate-pulse text-sm text-(--color-text-muted)">Loading predictions…</div>
        </div>
      )}

      {/* Error State - only show if NOT in off-season mode */}
      {error && !loading && !isOffSeason && (
        <div className="dashboard-wide rounded-lg border border-[rgba(239,68,68,0.35)] bg-[rgba(239,68,68,0.1)] p-4 text-sm text-red-200">
          <div className="font-extrabold">⚠️ {error}</div>
          <div className="mt-1 text-[rgba(226,232,240,0.8)]">Showing fallback data. Make sure the API server is running on {API_BASE_URL}</div>
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
            className="dashboard-wide"
            role="tabpanel"
            id="panel-current"
            aria-labelledby="tab-current"
            tabIndex={0}
          >
            {/* Hero: race context */}
            <section
              className="relative overflow-hidden rounded-lg border border-(--color-border) bg-(--color-surface)"
              style={{ backgroundImage: `url(/banner.jpg)`, backgroundSize: 'cover', backgroundPosition: 'center' }}
            >
              <div className="absolute inset-0 bg-[rgba(0,0,0,0.55)]" />
              <div className="relative p-5 sm:p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                  <div className="min-w-0">
                    <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">NEXT RACE</div>
                    <div className="mt-1 truncate text-2xl font-extrabold text-white">
                      {nextRace?.error ? 'Next race unavailable' : (nextRace?.raceName || 'TBD')}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-[rgba(226,232,240,0.85)]">
                      <span className="font-semibold">{nextRace?.error ? 'TBD' : (nextRace?.circuitName || 'TBD')}</span>
                      {nextRace?.error ? null : (
                        <span className="text-[rgba(148,163,184,0.95)]">{nextRace?.country || ''}</span>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] px-3 py-2">
                      <div className="text-[11px] font-bold tracking-wider text-(--color-text-muted)">RACE DAY</div>
                      <div className="mt-1 text-sm font-extrabold text-white">
                        {nextRaceDateStart ? nextRaceDateStart.toLocaleDateString('en-US', { month: 'short', day: '2-digit' }) : 'TBD'}
                      </div>
                    </div>
                    <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] px-3 py-2">
                      <div className="text-[11px] font-bold tracking-wider text-(--color-text-muted)">LOCAL TIME</div>
                      <div className="mt-1 text-sm font-extrabold text-white">
                        {nextRaceDateStart ? nextRaceDateStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }) : 'TBD'}
                      </div>
                    </div>
                    <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] px-3 py-2">
                      <div className="text-[11px] font-bold tracking-wider text-(--color-text-muted)">COUNTDOWN</div>
                      <div className="mt-1 text-sm font-extrabold text-white">{formatCountdown(nextRaceDateStart)}</div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] p-3">
                    <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">WINNER CONFIDENCE</div>
                    <div className="mt-1 text-sm font-bold text-white">
                      {winnerPrediction?.driver || '—'} · {typeof winnerPrediction?.percentage === 'number' ? `${winnerPrediction.percentage}%` : '—'}
                    </div>
                  </div>
                  <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] p-3">
                    <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">LAST 5 ACCURACY</div>
                    <div className="mt-1 text-sm font-bold text-white">
                      {typeof modelHealth.accuracy === 'number' ? `${modelHealth.accuracy}%` : '—'}
                      {modelHealth.total ? (
                        <span className="ml-2 text-xs font-semibold text-[rgba(148,163,184,0.95)]">({modelHealth.correct}/{modelHealth.total})</span>
                      ) : null}
                    </div>
                  </div>
                  <div className="rounded-md border border-[rgba(255,255,255,0.08)] bg-[rgba(0,0,0,0.25)] p-3">
                    <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">AVG CONFIDENCE (LAST 5)</div>
                    <div className="mt-1 text-sm font-bold text-white">
                      {typeof modelHealth.avgConfidence === 'number' ? `${modelHealth.avgConfidence}%` : '—'}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Main layout: left content + right rail */}
            <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
              <div className="flex flex-col gap-6">
                <WinnerPredictionCard
                  percentage={winnerPrediction?.percentage || 72}
                  driverName={winnerPrediction?.driver || 'NOR'}
                  teamColor="#ea580c"
                  headshotUrl={openF1DriverByAcronym.get((winnerPrediction?.driver || '').toUpperCase())?.headshotUrl}
                  fullName={openF1DriverByAcronym.get((winnerPrediction?.driver || '').toUpperCase())?.fullName}
                  confidence={winnerPrediction?.confidence || 'HIGH'}
                  confidenceColor={winnerPrediction?.confidence_color || '#f59e0b'}
                />

                <section className="rounded-lg border border-(--color-border) bg-(--color-surface) p-4">
                  <div className="text-xs font-extrabold tracking-wider text-(--color-text-muted)">RANKINGS</div>
                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {driverData.slice(1, 5).map((driver, idx) => {
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
                      )
                    })}
                  </div>

                  {driverData.length > 5 && (
                    <div className="mt-4 rounded-md border border-[rgba(255,255,255,0.06)] bg-[rgba(255,255,255,0.02)] p-3">
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">MORE DRIVERS</div>
                      <div className="mt-2 grid grid-cols-1 gap-2">
                        {driverData.slice(5, 10).map((driver, idx) => (
                          <div
                            key={`${driver?.name || 'driver'}-${driver?.position || idx}-row`}
                            className="flex items-center justify-between rounded-sm border border-[rgba(255,255,255,0.06)] bg-[rgba(0,0,0,0.2)] px-3 py-2"
                          >
                            <div className="min-w-0">
                              <div className="truncate text-sm font-semibold text-(--color-text-primary)">P{driver.position} · {driver.name}</div>
                              <div className="truncate text-xs text-(--color-text-muted)">{driver.team}</div>
                            </div>
                            <div className="ml-3 text-sm font-extrabold text-(--color-accent)">{driver.percentage}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </section>

                {/* Past 5 predictions table */}
                <RaceHistoryCard raceHistory={modelHealth.last5} />
              </div>

              <div className="flex flex-col gap-6">
                <section className="rounded-lg border border-(--color-border) bg-(--color-surface) p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-xs font-extrabold tracking-wider text-(--color-text-muted)">CIRCUIT INTELLIGENCE</div>
                    <button
                      type="button"
                      className="ui-focus-ring rounded-md border border-[rgba(255,255,255,0.12)] px-2 py-1 text-xs font-bold text-(--color-text-primary)"
                      onClick={() => setActiveTab('circuit')}
                    >
                      Open Map
                    </button>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">TRACK</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">{nextRace?.error ? 'TBD' : (nextRace?.circuitName || 'TBD')}</div>
                    </div>
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">COUNTRY</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">{nextRace?.error ? '—' : (nextRace?.country || '—')}</div>
                    </div>
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">RACE WINDOW</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">
                        {nextRaceDateStart ? nextRaceDateStart.toLocaleDateString('en-US', { month: 'short', day: '2-digit' }) : 'TBD'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">COUNTDOWN</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">{formatCountdown(nextRaceDateStart)}</div>
                    </div>
                  </div>
                </section>

                <section className="rounded-lg border border-(--color-border) bg-(--color-surface) p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-xs font-extrabold tracking-wider text-(--color-text-muted)">MODEL HEALTH</div>
                    <button
                      type="button"
                      className="ui-focus-ring rounded-md border border-[rgba(255,255,255,0.12)] px-2 py-1 text-xs font-bold text-(--color-text-primary)"
                      onClick={() => setActiveTab('mlops')}
                    >
                      Model Monitor
                    </button>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">LAST 5 ACCURACY</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">{typeof modelHealth.accuracy === 'number' ? `${modelHealth.accuracy}%` : '—'}</div>
                    </div>
                    <div>
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">AVG CONFIDENCE</div>
                      <div className="mt-1 font-semibold text-(--color-text-primary)">{typeof modelHealth.avgConfidence === 'number' ? `${modelHealth.avgConfidence}%` : '—'}</div>
                    </div>
                    <div className="col-span-2">
                      <div className="text-[11px] font-extrabold tracking-wider text-(--color-text-muted)">SAMPLE</div>
                      <div className="mt-1 text-sm text-(--color-text-secondary)">
                        {modelHealth.total ? `Computed from the last ${modelHealth.total} completed races.` : 'No completed races found yet.'}
                      </div>
                    </div>
                  </div>
                </section>
              </div>
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
