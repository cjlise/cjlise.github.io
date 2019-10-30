---
layout: archive
permalink: /categories/software-engineering/
title: "Software Engineering Posts by Tags"
author_profile: true
header:
   image: "/images/sky.jpg"

---

{% assign myposts = site.posts | where:"categories","software-engineering" %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in myposts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}


