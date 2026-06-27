import type { Metadata } from "next";
import { Geist, Geist_Mono, Playfair_Display } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Providers from "@/components/Providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const playfairDisplay = Playfair_Display({
  variable: "--font-playfair-display",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
});

export const metadata: Metadata = {
  title: "Kellan McIntosh — Data Science & ML Engineer",
  description:
    "Portfolio of Kellan McIntosh, a Data Science and Machine Learning Engineering student at UMass Amherst. Projects in Python, PyTorch, scikit-learn, and Next.js.",
  openGraph: {
    title: "Kellan McIntosh — Data Science & ML Engineer",
    description:
      "Portfolio of Kellan McIntosh, a Data Science and Machine Learning Engineering student at UMass Amherst.",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "Kellan McIntosh — Data Science & ML Engineer",
    description:
      "Portfolio of Kellan McIntosh, a Data Science and Machine Learning Engineering student at UMass Amherst.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${playfairDisplay.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Providers>
          <Navbar />
          {children}
        </Providers>
      </body>
    </html>
  );
}
