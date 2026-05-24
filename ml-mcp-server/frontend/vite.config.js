import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8765",
      "/tools": "http://127.0.0.1:8765",
      "/experiments": "http://127.0.0.1:8765",
      "/call": "http://127.0.0.1:8765",
      "/rpc": "http://127.0.0.1:8765",
    },
  },
});

