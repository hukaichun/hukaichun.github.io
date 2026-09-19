// Minimal tag filter for the post list on the home page.
// The active tag lives in the URL hash (e.g. /#rerank), so tag links on
// post pages can deep-link into a filtered home page.
(function () {
  try {
    var tagLinks = document.querySelectorAll("[data-tag-filter]");
    var cards = document.querySelectorAll("[data-post-tags]");
    if (!cards.length) return;

    function currentTag() {
      return decodeURIComponent((location.hash || "").replace(/^#/, ""));
    }

    function applyFilter() {
      var tag = currentTag();
      cards.forEach(function (card) {
        var tags = (card.getAttribute("data-post-tags") || "").split(",");
        card.style.display = !tag || tags.indexOf(tag) !== -1 ? "" : "none";
      });
      tagLinks.forEach(function (link) {
        link.classList.toggle("active", link.getAttribute("data-tag-filter") === tag);
      });
    }

    // Clicking the already-active tag clears the filter.
    tagLinks.forEach(function (link) {
      link.addEventListener("click", function (e) {
        var tag = link.getAttribute("data-tag-filter");
        if (tag && tag === currentTag()) {
          e.preventDefault();
          history.replaceState(null, "", location.pathname);
          applyFilter();
        }
      });
    });

    window.addEventListener("hashchange", applyFilter);
    applyFilter();
  } catch (e) {
    // DOM quirks should never break the page.
  }
})();
