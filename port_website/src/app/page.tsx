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
      {/* Hero */}
      <section className="flex min-h-screen flex-col items-center justify-center">
        <h1 className="text-5xl font-semibold tracking-tight text-text-primary">
          Kellan McIntosh
        </h1>
        <p className="mt-4 text-lg text-text-secondary">
          Data science &amp; ML portfolio
        </p>
      </section>

      {/* Stub sections */}
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
