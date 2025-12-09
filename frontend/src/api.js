// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
const OPENF1_API = 'https://api.openf1.org/v1';

// Simple cache to reduce API calls and avoid rate limiting
const cache = {
    nextRace: null,
    drivers: null,
    timestamp: 0
};
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

function isCacheValid() {
    return Date.now() - cache.timestamp < CACHE_DURATION;
}

/**
 * Fetch race history from backend - last 5 races with predictions vs actual results
 */
export async function fetchRaceHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/race-history`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch race history');
        }

        return result.data || [];
    } catch (error) {
        console.error('Error fetching race history:', error);
        throw error;
    }
}

/**
 * Fetch qualifying data from API
 * @param {Object} params - Race parameters
 * @param {string} params.race_key - Race key identifier (optional)
 * @param {number} params.race_year - Race year (optional)
 * @param {string} params.circuit - Circuit name (optional)
 * @param {string} params.event - Event name (optional)
 * @returns {Promise<Array>} Qualifying data array
 */
export async function getQualifying({ race_key, race_year, circuit, event }) {
    try {
        const params = new URLSearchParams();
        if (race_key) params.set("race_key", race_key);
        if (race_year) params.set("race_year", race_year);
        if (circuit) params.set("circuit", circuit);
        if (event) params.set("event", event);

        const url = `${API_BASE_URL.replace(/\/$/, "")}/api/qualifying?${params.toString()}`;
        const response = await fetch(url);

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`Qualifying fetch failed: ${text || response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || "No qualifying data available");
        }

        return result.qualifying;
    } catch (error) {
        console.error('Error fetching qualifying:', error);
        throw error;
    }
}

/**
 * Fetch predictions by first getting qualifying data, then calling predict
 * @param {Object} raceMeta - Race metadata
 * @param {string} raceMeta.race_key - Race key identifier
 * @param {number} raceMeta.race_year - Race year
 * @param {string} raceMeta.circuit - Circuit name
 * @param {string} raceMeta.event - Event name
 * @returns {Promise<Object>} Prediction results
 */
export async function fetchAndPredict(raceMeta) {
    try {
        // First, fetch qualifying data
        const qualifying = await getQualifying(raceMeta);

        // Then call predict with race metadata + qualifying
        const payload = { ...raceMeta, qualifying };
        const predictions = await fetchCustomPredictions(payload);

        return predictions;
    } catch (error) {
        console.error("Auto fetch/predict failed:", error);
        throw error;
    }
}

/**
 * Fetch predictions for São Paulo Grand Prix
 */
export async function fetchSaoPauloPredictions() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict/sao-paulo`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        // The backend returns race_history and next_race, not full_predictions
        const nextRace = result.next_race;
        const raceHistory = result.race_history || [];

        // If no next race, show the most recent race with "waiting for next race" message
        if (!nextRace) {
            // Get most recent race from history
            const mostRecentRace = raceHistory && raceHistory.length > 0
                ? raceHistory[0]
                : null;

            return {
                isSeasonEnded: true,
                mostRecentRace: mostRecentRace,
                winner_prediction: {
                    driver: 'TBD',
                    team: 'Unknown',
                    percentage: 0,
                    confidence: 'N/A'
                },
                top3_prediction: [],
                full_predictions: [],
                race_history: raceHistory,
                message: 'Waiting for next race in 2026 season...'
            };
        }

        // Normal case: next race exists
        return {
            isSeasonEnded: false,
            winner_prediction: {
                driver: nextRace.predicted_winner,
                team: nextRace.team || 'Unknown',
                percentage: nextRace.predicted_confidence,
                confidence: 'HIGH'
            },
            top3_prediction: nextRace.predicted_top3 || [],
            full_predictions: nextRace.full_predictions || [],
            race_history: raceHistory
        };
    } catch (error) {
        console.error('Error fetching predictions:', error);
        throw error;
    }
}

/**
 * Fetch predictions for a custom race with qualifying data
 */
export async function fetchCustomPredictions(raceData) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(raceData)
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        return result.data;
    } catch (error) {
        console.error('Error fetching custom predictions:', error);
        throw error;
    }
}

/**
 * Check API health
 */
export async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        const result = await response.json();
        return result.status === 'online';
    } catch (error) {
        console.error('API health check failed:', error);
        return false;
    }
}

/**
 * Transform API predictions to frontend format
 */
export function transformPredictionsToDriverData(predictions) {
    if (!predictions || !predictions.full_predictions) {
        console.warn('No full_predictions in predictions object:', predictions);
        return [];
    }

    const teamColors = {
        'McLaren': '#ea580c',
        'Mercedes': '#64748b',
        'Ferrari': '#dc2626',
        'Red Bull': '#1e3a8a',
        'Racing Bulls': '#4338ca',
        'Haas': '#6b7280',
        'Alpine': '#0ea5e9',
        'Kick Sauber': '#10b981',
        'Aston Martin': '#059669',
        'Williams': '#0284c7',
    };

    return predictions.full_predictions.slice(0, 6).map((pred, idx) => ({
        name: pred.driver,
        team: pred.team,
        percentage: pred.percentage || pred.win_prob || 0,
        teamColor: teamColors[pred.team] || '#64748b',
        position: idx + 1,
        points: 0, // Not available from API
        predictedWin: pred.p_win || pred.win_prob || 0,
        confidence: pred.confidence || 'MEDIUM',
        confidenceColor: pred.confidence_color || '#f59e0b'
    }));
}

/**
 * Fetch next race information from OpenF1 API
 */
export async function fetchNextRace() {
    // Return cached data if still valid
    if (isCacheValid() && cache.nextRace) {
        console.log('Using cached next race data');
        return cache.nextRace;
    }

    try {
        // Use backend API to get 2025 data from FastF1
        const response = await fetch(`${API_BASE_URL}/api/next-race`);

        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch next race');
        }

        const raceData = result.race || result;

        // Cache and return the result
        cache.nextRace = {
            sessionName: 'Race',
            raceName: raceData.event_name,
            country: raceData.circuit,
            location: raceData.circuit,
            circuitName: raceData.circuit,
            circuitKey: raceData.circuit?.toLowerCase().replace(/\s+/g, '-'),
            circuitImage: result.circuit_image_url,
            dateStart: new Date(raceData.date),
            dateEnd: new Date(raceData.date),
            year: raceData.year,
            sessionKey: 'race'
        };
        cache.timestamp = Date.now();

        console.log('✓ Next race loaded:', cache.nextRace.raceName, 'on', raceData.date);
        return cache.nextRace;
    } catch (error) {
        console.error('Error fetching next race from backend:', error);
        return getFallbackRaceData();
    }
}

function getFallbackRaceData() {
    console.warn('Using fallback race data');
    return {
        raceName: 'Abu Dhabi Grand Prix',
        country: 'United Arab Emirates',
        location: 'Abu Dhabi',
        circuitName: 'Yas Marina Circuit',
        circuitKey: 'yas-marina',
        circuitImage: 'https://media.formula1.com/image/upload/f_auto,c_limit,w_1440,q_auto/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Abu_Dhabi_Circuit',
        dateStart: new Date('2024-12-08T13:00:00Z'),
        dateEnd: new Date('2024-12-08T15:00:00Z'),
        year: 2024,
        sessionKey: 'latest'
    };
}

/**
 * Fetch driver information including images from OpenF1
 */
export async function fetchDriverInfo(driverNumber) {
    try {
        const response = await fetch(`${OPENF1_API}/drivers?driver_number=${driverNumber}&session_key=latest`);

        if (!response.ok) {
            throw new Error(`OpenF1 driver API error: ${response.status}`);
        }

        const drivers = await response.json();
        if (drivers.length === 0) {
            return null;
        }

        const driver = drivers[0];
        // Use higher quality image from F1 CDN - try both 2024 and 2025 formats
        const acronym = driver.name_acronym?.toUpperCase();
        const highResHeadshot = driver.headshot_url || `https://media.formula1.com/image/upload/f_auto,c_limit,w_1440,q_auto/content/dam/fom-website/drivers/2024Drivers/${acronym}.jpg`;

        return {
            fullName: driver.full_name,
            nameAcronym: driver.name_acronym,
            teamName: driver.team_name,
            teamColour: driver.team_colour ? `#${driver.team_colour}` : null,
            headshotUrl: highResHeadshot,
            countryCode: driver.country_code,
            driverNumber: driver.driver_number
        };
    } catch (error) {
        console.error('Error fetching driver info from OpenF1:', error);
        return null;
    }
}

/**
 * Fetch all drivers for current session
 */
export async function fetchCurrentDrivers() {
    // Return cached data if still valid
    if (isCacheValid() && cache.drivers) {
        console.log('Using cached drivers data');
        return cache.drivers;
    }

    try {
        const response = await fetch(`${OPENF1_API}/drivers?session_key=latest`);

        if (!response.ok) {
            // If rate limited, return cached or empty
            if (response.status === 429) {
                console.warn('OpenF1 rate limit hit for drivers');
                return cache.drivers || [];
            }
            throw new Error(`OpenF1 drivers API error: ${response.status}`);
        }

        const drivers = await response.json();
        const driverData = drivers.map(driver => {
            // Use higher quality image from F1 CDN - try both formats
            const acronym = driver.name_acronym?.toUpperCase();
            const highResHeadshot = driver.headshot_url || `https://media.formula1.com/image/upload/f_auto,c_limit,w_1440,q_auto/content/dam/fom-website/drivers/2024Drivers/${acronym}.jpg`;

            return {
                fullName: driver.full_name,
                nameAcronym: driver.name_acronym,
                teamName: driver.team_name,
                teamColour: driver.team_colour ? `#${driver.team_colour}` : null,
                headshotUrl: highResHeadshot,
                countryCode: driver.country_code,
                driverNumber: driver.driver_number
            };
        });

        // Cache the result
        cache.drivers = driverData;
        cache.timestamp = Date.now();

        return driverData;
    } catch (error) {
        console.error('Error fetching current drivers from OpenF1:', error);
        return cache.drivers || [];
    }
}

/**
 * Fetch live race updates from OpenF1
 */
export async function fetchLiveRaceUpdates() {
    try {
        const now = new Date();

        // First, try to get ongoing race session
        let response = await fetch(`${OPENF1_API}/sessions?session_name=Race&date_start<=${now.toISOString()}&date_end>=${now.toISOString()}&limit=1`);

        if (!response.ok) {
            throw new Error(`OpenF1 API error: ${response.status}`);
        }

        let sessions = await response.json();

        if (sessions.length > 0) {
            // Live race found - fetch position data
            const sessionKey = sessions[0].session_key;
            const posResponse = await fetch(`${OPENF1_API}/position?session_key=${sessionKey}&date>=${new Date(Date.now() - 300000).toISOString()}`);

            if (posResponse.ok) {
                const positions = await posResponse.json();
                return {
                    isLive: true,
                    session: sessions[0],
                    positions: positions.slice(-20) // Last 20 position updates
                };
            }
        }

        // No live race - get most recent race
        response = await fetch(`${OPENF1_API}/sessions?session_name=Race&date_start<=${now.toISOString()}&limit=1`);
        sessions = await response.json();

        if (sessions.length === 0) {
            // Try 2024 season as fallback
            response = await fetch(`${OPENF1_API}/sessions?session_name=Race&year=2024&limit=1`);
            sessions = await response.json();
        }

        if (sessions.length > 0) {
            const sessionKey = sessions[0].session_key;
            // Try to get some position data from the race
            try {
                const posResponse = await fetch(`${OPENF1_API}/position?session_key=${sessionKey}`);
                if (posResponse.ok) {
                    const positions = await posResponse.json();
                    return {
                        isLive: false,
                        session: sessions[0],
                        positions: positions.slice(-10)
                    };
                }
            } catch (e) {
                console.log('No position data available for recent race');
            }

            return {
                isLive: false,
                session: sessions[0],
                positions: []
            };
        }

        return { isLive: false, session: null, positions: [] };
    } catch (error) {
        console.error('Error fetching live race updates:', error);
        return { isLive: false, session: null, positions: [] };
    }
}

/**
 * Fetch driver standings from OpenF1
 */
export async function fetchDriverStandings() {
    try {
        // Use backend API to get 2025 standings with actual and predicted points
        console.log('fetchDriverStandings: Calling', `${API_BASE_URL}/api/driver-standings`)
        const response = await fetch(`${API_BASE_URL}/api/driver-standings`);

        console.log('fetchDriverStandings: Response status', response.status)
        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const result = await response.json();
        console.log('fetchDriverStandings: Result', result)

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch driver standings');
        }

        // Map backend data to frontend format
        const drivers = result.data || [];
        console.log('fetchDriverStandings: Mapped', drivers.length, 'drivers')

        return drivers.map((driver) => ({
            position: driver.position,
            driverName: driver.driverName,
            teamName: driver.teamName,
            points: driver.points,
            predictedPoints: driver.predictedPoints,
            headshotUrl: driver.headshotUrl,
            teamColor: driver.teamColor || '#64748b'
        }));
    } catch (error) {
        console.error('Error fetching driver standings:', error);
        // Fallback: try OpenF1 as backup
        try {
            const response = await fetch(`${OPENF1_API}/drivers?session_key=latest`);
            if (response.ok) {
                const drivers = await response.json();
                return drivers.map((driver, idx) => ({
                    position: idx + 1,
                    driverName: driver.full_name,
                    teamName: driver.team_name,
                    points: 0,
                    predictedPoints: 0,
                    headshotUrl: driver.headshot_url,
                    teamColor: driver.team_colour ? `#${driver.team_colour}` : '#64748b'
                }));
            }
        } catch (fallbackErr) {
            console.error('Fallback OpenF1 also failed:', fallbackErr);
        }
        return [];
    }
}

/**
 * Fetch constructor/team standings from backend
 */
export async function fetchConstructorStandings() {
    try {
        // Use backend API to get 2025 standings with actual and predicted points
        console.log('fetchConstructorStandings: Calling', `${API_BASE_URL}/api/constructor-standings`);
        const response = await fetch(`${API_BASE_URL}/api/constructor-standings`);

        console.log('fetchConstructorStandings: Response status', response.status);
        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const result = await response.json();
        console.log('fetchConstructorStandings: Result', result);

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch constructor standings');
        }

        // Map backend data to frontend format
        const constructors = result.data || [];
        console.log('fetchConstructorStandings: Mapped', constructors.length, 'constructors');

        return constructors.map((constructor) => ({
            position: constructor.position,
            constructorName: constructor.constructorName,
            points: constructor.points,
            predictedPoints: constructor.predictedPoints,
            wins: constructor.wins,
            teamColor: constructor.teamColor || '#64748b'
        }));
    } catch (error) {
        console.error('Error fetching constructor standings:', error);
        return [];
    }
}

/**
 * Fetch next race from Ergast via backend
 */
export async function fetchNextRaceFromErgast() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/next-race`);

        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const data = await response.json();
        if (!data.success || !data.race) {
            throw new Error('No race data available');
        }

        const race = data.race;
        return {
            raceName: race.event || 'Unknown GP',
            country: race.country,
            location: race.circuit,
            circuitName: race.circuit,
            circuitKey: race.race_key,
            circuitImage: data.circuit_image_url ? `${API_BASE_URL}${data.circuit_image_url}` : null,
            dateStart: new Date(race.date + 'T' + (race.time || '00:00:00')),
            dateEnd: new Date(race.date + 'T' + (race.time || '00:00:00')),
            year: race.race_year,
            sessionKey: race.race_key
        };
    } catch (error) {
        console.error('Error fetching next race from Ergast:', error);
        throw error;
    }
}

/**
 * Fetch standings from Ergast via backend
 */
export async function fetchStandingsFromErgast(season = 'current') {
    try {
        const response = await fetch(`${API_BASE_URL}/api/standings?season=${season}`);

        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error('Failed to fetch standings');
        }

        return data.standings;
    } catch (error) {
        console.error('Error fetching standings from Ergast:', error);
        return { drivers: [], constructors: [] };
    }
}

/**
 * Fetch driver image from backend/Wikipedia cache
 */
export async function fetchDriverImageFromBackend(driverName) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/driver-image?name=${encodeURIComponent(driverName)}`);

        if (!response.ok) {
            return null;
        }

        const data = await response.json();
        if (data.success && data.image_url) {
            return `${API_BASE_URL}${data.image_url}`;
        }

        return null;
    } catch (error) {
        console.error('Error fetching driver image:', error);
        return null;
    }
}

/**
 * Fetch latest race circuit data with top 6 drivers
 */
export async function fetchLatestRaceCircuit() {
    try {
        console.log('fetchLatestRaceCircuit: Calling', `${API_BASE_URL}/api/latest-race-circuit`);
        const response = await fetch(`${API_BASE_URL}/api/latest-race-circuit`);

        console.log('fetchLatestRaceCircuit: Response status', response.status);
        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const result = await response.json();
        console.log('fetchLatestRaceCircuit: Result', result);

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch latest race circuit');
        }

        return result.data;
    } catch (error) {
        console.error('Error fetching latest race circuit:', error);
        return null;
    }
}

export default {
    fetchSaoPauloPredictions,
    fetchCustomPredictions,
    checkAPIHealth,
    transformPredictionsToDriverData,
    getQualifying,
    fetchAndPredict,
    fetchRaceHistory,
    fetchNextRace,
    fetchDriverInfo,
    fetchCurrentDrivers,
    fetchLiveRaceUpdates,
    fetchDriverStandings,
    fetchConstructorStandings,
    fetchNextRaceFromErgast,
    fetchStandingsFromErgast,
    fetchDriverImageFromBackend,
    fetchLatestRaceCircuit
};
