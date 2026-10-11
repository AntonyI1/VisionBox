const CACHE = 'visionbox-static-v4';
const OFFLINE_PAGE = '/static/offline.html';
const PRECACHE = [
    OFFLINE_PAGE,
    '/static/style.css',
    '/static/app.js',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/manifest.webmanifest',
];

self.addEventListener('install', event => {
    // /static/* is served with max-age=3600, so a plain addAll() could precache the
    // browser's copy of the old app.js into the new cache; fetch past the HTTP cache.
    event.waitUntil(
        caches.open(CACHE)
            .then(cache => cache.addAll(PRECACHE.map(url => new Request(url, { cache: 'reload' }))))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys()
            .then(keys => Promise.all(
                keys.filter(k => k !== CACHE).map(k => caches.delete(k))
            ))
            .then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', event => {
    const req = event.request;
    if (req.method !== 'GET') return;
    const url = new URL(req.url);
    if (url.origin !== self.location.origin) return;

    // Never cache: API (streams, snapshots, events, media) and auth pages.
    if (url.pathname.startsWith('/api/') ||
        url.pathname === '/login' || url.pathname === '/logout') return;

    // Navigations (dashboard shell, login redirects) are network-first so auth
    // and fresh HTML always win; the offline page is the last resort.
    if (req.mode === 'navigate') {
        event.respondWith(
            fetch(req).catch(() => caches.match(OFFLINE_PAGE))
        );
        return;
    }

    // Static assets (CSS/JS/icons) and the manifest: cache-first.
    if (url.pathname.startsWith('/static/') ||
        url.pathname === '/manifest.webmanifest') {
        event.respondWith(
            caches.match(req).then(hit => hit || fetch(req).then(resp => {
                if (resp.ok) {
                    const copy = resp.clone();
                    caches.open(CACHE).then(cache => cache.put(req, copy));
                }
                return resp;
            }))
        );
    }
});
