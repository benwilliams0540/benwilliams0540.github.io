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

  // ---- Screenshot carousel ----
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelectorAll("[data-carousel]").forEach((carousel) => {
    const track = carousel.querySelector("[data-carousel-track]");
    const slides = Array.from(carousel.querySelectorAll("[data-carousel-slide]"));
    const dots = Array.from(carousel.querySelectorAll("[data-carousel-dot]"));
    const previousButton = carousel.querySelector("[data-carousel-prev]");
    const nextButton = carousel.querySelector("[data-carousel-next]");
    const toggleButton = carousel.querySelector("[data-carousel-toggle]");
    const toggleLabel = carousel.querySelector("[data-carousel-toggle-label]");
    const autoplayDelay = 8000;
    let activeIndex = 0;
    let autoplayPaused = reduceMotion;
    let scrollSettleTimer;
    let timer;

    if (!track || slides.length < 2 || dots.length !== slides.length) {
      return;
    }

    const setActiveDot = (index) => {
      dots.forEach((dot, dotIndex) => {
        const isActive = dotIndex === index;
        dot.classList.toggle("is-active", isActive);
        dot.setAttribute("aria-current", isActive ? "true" : "false");
      });
    };

    const updateToggle = () => {
      if (!toggleButton) return;
      const label = autoplayPaused ? "Play slideshow" : "Pause slideshow";
      toggleButton.classList.toggle("is-paused", autoplayPaused);
      toggleButton.setAttribute("aria-label", label);
      toggleButton.setAttribute("aria-pressed", autoplayPaused ? "true" : "false");
      toggleButton.setAttribute("title", label);
      if (toggleLabel) toggleLabel.textContent = label;
    };

    const stopAutoplay = () => {
      window.clearInterval(timer);
    };

    const goToSlide = (index, behavior = "smooth") => {
      activeIndex = (index + slides.length) % slides.length;
      track.scrollTo({
        left: track.clientWidth * activeIndex,
        behavior,
      });
      setActiveDot(activeIndex);
    };

    const schedule = () => {
      stopAutoplay();
      if (autoplayPaused) return;
      timer = window.setInterval(() => {
        goToSlide(activeIndex + 1);
      }, autoplayDelay);
    };

    dots.forEach((dot, index) => {
      dot.addEventListener("click", () => {
        goToSlide(index);
        schedule();
      });
    });

    previousButton?.addEventListener("click", () => {
      goToSlide(activeIndex - 1);
      schedule();
    });

    nextButton?.addEventListener("click", () => {
      goToSlide(activeIndex + 1);
      schedule();
    });

    toggleButton?.addEventListener("click", () => {
      autoplayPaused = !autoplayPaused;
      updateToggle();
      schedule();
    });

    track.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goToSlide(activeIndex - 1);
        schedule();
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        goToSlide(activeIndex + 1);
        schedule();
      }
    });

    track.addEventListener("scroll", () => {
      window.clearTimeout(scrollSettleTimer);
      scrollSettleTimer = window.setTimeout(() => {
        const nextIndex = Math.round(track.scrollLeft / Math.max(track.clientWidth, 1));
        if (nextIndex !== activeIndex && nextIndex >= 0 && nextIndex < slides.length) {
          activeIndex = nextIndex;
          setActiveDot(activeIndex);
        }
      }, 120);
    });

    carousel.addEventListener("mouseenter", stopAutoplay);
    carousel.addEventListener("mouseleave", schedule);
    carousel.addEventListener("focusin", stopAutoplay);
    carousel.addEventListener("focusout", schedule);

    setActiveDot(activeIndex);
    updateToggle();
    schedule();
  });
})();
