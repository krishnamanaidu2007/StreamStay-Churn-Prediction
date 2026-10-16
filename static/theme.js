(function () {
    "use strict";

    var storageKey = "streamstay-theme";
    var root = document.documentElement;

    function savedTheme() {
        try {
            return localStorage.getItem(storageKey) === "light" ? "light" : "dark";
        } catch (error) {
            return "dark";
        }
    }

    function applyTheme(theme) {
        var isLight = theme === "light";
        root.dataset.theme = isLight ? "light" : "dark";

        var toggle = document.querySelector("[data-theme-toggle]");
        if (toggle) {
            toggle.setAttribute("aria-pressed", String(isLight));
            toggle.setAttribute("aria-label", isLight ? "Switch to dark mode" : "Switch to light mode");
            toggle.setAttribute("title", isLight ? "Switch to dark mode" : "Switch to light mode");
        }
    }

    applyTheme(savedTheme());

    document.addEventListener("DOMContentLoaded", function () {
        applyTheme(savedTheme());
        var toggle = document.querySelector("[data-theme-toggle]");
        if (!toggle) return;

        toggle.addEventListener("click", function () {
            var nextTheme = root.dataset.theme === "light" ? "dark" : "light";
            try {
                localStorage.setItem(storageKey, nextTheme);
            } catch (error) {
                // The chosen theme remains active for this page when storage is unavailable.
            }
            applyTheme(nextTheme);
        });
    });
}());
