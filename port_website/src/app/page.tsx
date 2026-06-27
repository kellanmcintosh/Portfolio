import Hero from "@/components/Hero";

const SECTIONS = [
  { id: "projects", label: "Projects" },
  { id: "about", label: "About" },
  { id: "resume", label: "Resume" },
  { id: "grades", label: "Grades" },
  { id: "contact", label: "Contact" },
];

export default function Home() {
  return (
    <main>
      <Hero />
      {SECTIONS.map(({ id, label }) => (
        <section
          key={id}
          id={id}
          className="flex min-h-screen items-center justify-center border-t border-border"
        >
          <h2 className="text-3xl font-semibold text-text-primary">{label}</h2>
        </section>
      ))}
    </main>
  );
}
