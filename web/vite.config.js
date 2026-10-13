import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The build lands inside the Python package so the wheel ships the dashboard.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: "../moescope/server/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:7878" } },
});
