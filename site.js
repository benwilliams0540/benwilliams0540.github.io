// Portfolio interactions: reveal-on-scroll, lazy splat-viewer launch, viewer tabs.
(() => {
  "use strict";

  // ---- Reveal on scroll ----
  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && reveals.length) {
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.04 }
    );
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }

  // ---- Lazy-launch the splat viewer ----
  document.querySelectorAll("[data-launch]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-launch");
      const stage = document.getElementById("stage-" + id);
      if (!stage || stage.dataset.loaded === "1") return;

      const src = stage.getAttribute("data-src");
      const loading = document.getElementById("loading-" + id);
      if (loading) loading.classList.add("show");

      const iframe = document.createElement("iframe");
      iframe.setAttribute("title", "Interactive Gaussian splat viewer");
      iframe.setAttribute("allow", "fullscreen; xr-spatial-tracking; accelerometer; gyroscope");
      iframe.setAttribute("loading", "eager");
      iframe.addEventListener("load", () => {
        if (loading) loading.classList.remove("show");
      });
      iframe.src = src;

      stage.appendChild(iframe);
      stage.dataset.loaded = "1";

      const scrim = stage.querySelector(".splat-scrim");
      if (scrim) scrim.style.display = "none";
      const poster = stage.querySelector(".splat-poster");
      if (poster) poster.style.opacity = "0";
    });
  });

  // ---- Viewer scene tabs ----
  document.querySelectorAll(".viewer-tabs").forEach((tabs) => {
    const block = tabs.closest(".viewer-block");
    if (!block) return;
    tabs.querySelectorAll("[data-tab]").forEach((tab) => {
      tab.addEventListener("click", () => {
        const id = tab.getAttribute("data-tab");
        tabs.querySelectorAll("[data-tab]").forEach((t) => {
          t.setAttribute("aria-selected", t === tab ? "true" : "false");
        });
        block.querySelectorAll(".splat-scene").forEach((scene) => {
          scene.classList.toggle("is-hidden", scene.getAttribute("data-scene") !== id);
        });
      });
    });
  });
})();
