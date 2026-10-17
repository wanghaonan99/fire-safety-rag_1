/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 后端在 Render，前端在 Vercel，跨域由后端 CORS 处理，无需 rewrites
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000',
  },
};

export default nextConfig;
