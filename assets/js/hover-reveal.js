(function () {
  // Random rainbow color on hover for all links and .reveal-hover elements
  function randomColor() {
    var hue = Math.floor(Math.random() * 360);
    return 'hsl(' + hue + ', 80%, 50%)';
  }

  function addRainbowHover(el) {
    el.addEventListener('mouseenter', function () {
      el.style.color = randomColor();
    });
    el.addEventListener('mouseleave', function () {
      el.style.color = '';
    });
  }

  document.querySelectorAll('a, .reveal-hover').forEach(addRainbowHover);

  // Hover-reveal image follow — only on hover-capable devices
  if (!window.matchMedia('(hover: hover)').matches) return;

  var items = document.querySelectorAll('.reveal-hover[data-hover-img]');
  if (!items.length) return;

  // Create the floating image container
  var container = document.createElement('div');
  container.id = 'hover-reveal';
  var img = document.createElement('img');
  container.appendChild(img);
  document.body.appendChild(container);

  // Preload images
  var preloaded = {};
  items.forEach(function (el) {
    var src = el.getAttribute('data-hover-img');
    if (src && !preloaded[src]) {
      var preImg = new Image();
      preImg.src = src;
      preloaded[src] = true;
    }
  });

  function position(e) {
    var x = e.clientX + 20;
    var y = e.clientY + 20;

    var rect = container.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) {
      x = e.clientX - rect.width - 20;
    }
    if (y + rect.height > window.innerHeight) {
      y = e.clientY - rect.height - 20;
    }

    container.style.left = x + 'px';
    container.style.top = y + 'px';
  }

  items.forEach(function (el) {
    var src = el.getAttribute('data-hover-img');

    el.addEventListener('mouseenter', function (e) {
      img.src = src;
      position(e);
      container.classList.add('visible');
    });

    el.addEventListener('mousemove', position);

    el.addEventListener('mouseleave', function () {
      container.classList.remove('visible');
    });
  });
})();
