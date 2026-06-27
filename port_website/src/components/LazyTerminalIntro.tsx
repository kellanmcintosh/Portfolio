"use client";

import dynamic from "next/dynamic";

const TerminalIntro = dynamic(() => import("./TerminalIntro"), { ssr: false });

export default function LazyTerminalIntro() {
  return <TerminalIntro />;
}
