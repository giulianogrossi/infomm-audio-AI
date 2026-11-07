(() => {
  function startDemo(frame) {
    if (frame.dataset.demoSrc) {
      frame.src = frame.dataset.demoSrc;
      delete frame.dataset.demoSrc;
    }
  }
  function initDemos() {
    const frames = document.querySelectorAll("iframe[data-demo-src]");
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting && entry.target.clientWidth > 0 && entry.target.clientHeight > 0) {
            startDemo(entry.target);
            observer.unobserve(entry.target);
          }
        });
      }, {rootMargin: "150px"});
      frames.forEach(frame => observer.observe(frame));
    }
    document.addEventListener("shown.bs.collapse", event => {
      event.target.querySelectorAll("iframe[data-demo-src]").forEach(startDemo);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initDemos);
  else initDemos();
})();
