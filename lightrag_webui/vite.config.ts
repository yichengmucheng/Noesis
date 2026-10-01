// vite.config.ts
import { defineConfig, loadEnv } from 'vite'
import path from 'node:path'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// 仅在源码中使用路径别名，config 本身不再 import '@/...'
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '') // 允许读取 VITE_* 变量

  // 优先使用 .env 中的 VITE_BASE_URL，其次用之前你注释的默认 '/webui/'
  const baseFromEnv = env.VITE_BASE_URL && env.VITE_BASE_URL.trim() !== '' ? env.VITE_BASE_URL : '/webui/'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    base: baseFromEnv, // ← 不再从 src/lib/constants 读取
    build: {
      outDir: path.resolve(__dirname, '../lightrag/api/webui'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            'react-vendor': ['react', 'react-dom', 'react-router-dom'],
            'graph-vendor': ['sigma', 'graphology', '@react-sigma/core'],
            'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-popover', '@radix-ui/react-select', '@radix-ui/react-tabs'],
            'utils-vendor': ['axios', 'i18next', 'zustand', 'clsx', 'tailwind-merge'],
            'feature-graph': ['./src/features/GraphViewer'],
            'feature-documents': ['./src/features/DocumentManager'],
            'feature-retrieval': ['./src/features/RetrievalTesting'],
            'mermaid-vendor': ['mermaid'],
            'markdown-vendor': [
              'react-markdown',
              'rehype-react',
              'remark-gfm',
              'remark-math',
              'react-syntax-highlighter'
            ]
          },
          chunkFileNames: 'assets/[name]-[hash].js',
          entryFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    server: {
      proxy:
        env.VITE_API_PROXY === 'true' && env.VITE_API_ENDPOINTS
          ? Object.fromEntries(
            env.VITE_API_ENDPOINTS.split(',').map((endpoint) => [
              endpoint,
              {
                target: env.VITE_BACKEND_URL || 'http://localhost:9621',
                changeOrigin: true,
                rewrite:
                  endpoint === '/api'
                    ? (p: string) => p.replace(/^\/api/, '')
                    : endpoint === '/docs' || endpoint === '/openapi.json'
                      ? (p: string) => p
                      : undefined
              }
            ])
          )
          : {}
    }
  }
})
