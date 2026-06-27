"use client";

import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

export default function ResumePDFViewer() {
  const [numPages, setNumPages] = useState(0);
  const [width, setWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={containerRef}>
      <Document
        file="/resume.pdf"
        onLoadSuccess={({ numPages }) => setNumPages(numPages)}
        loading={
          <div
            className="w-full animate-pulse bg-surface"
            style={{ aspectRatio: "8.5 / 11" }}
          />
        }
      >
        {width > 0 &&
          Array.from({ length: numPages }, (_, i) => (
            <div
              key={i + 1}
              className={i > 0 ? "mt-6" : ""}
              style={{
                boxShadow:
                  "0 2px 8px rgba(0,0,0,0.35), 0 16px 48px rgba(0,0,0,0.3)",
              }}
            >
              <Page
                pageNumber={i + 1}
                width={width}
                renderTextLayer
                renderAnnotationLayer
              />
            </div>
          ))}
      </Document>
    </div>
  );
}
