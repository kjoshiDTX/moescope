const cache = new Map();

export async function getJSON(path) {
  if (cache.has(path)) return cache.get(path);
  const promise = fetch(path).then(async (res) => {
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });
  cache.set(path, promise);
  promise.catch(() => cache.delete(path));
  return promise;
}

export const runPath = (name) => `/api/runs/${name.split("/").map(encodeURIComponent).join("/")}`;
