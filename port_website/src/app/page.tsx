import Hero from "@/components/Hero";
import LazyTerminalIntro from "@/components/LazyTerminalIntro";
import ProjectsSection from "@/components/ProjectsSection";
import AboutSection from "@/components/AboutSection";
import GradesSection from "@/components/GradesSection";
import ResumeSection from "@/components/ResumeSection";
import ContactSection from "@/components/ContactSection";

export default function Home() {
  return (
    <main>
      <LazyTerminalIntro />
      <Hero />
      <ProjectsSection />
      <AboutSection />
      <ResumeSection />
      <GradesSection />
      <ContactSection />
    </main>
  );
}
