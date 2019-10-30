---
layout: category
permalink: /categories/machine-learning/
title: "Machine Learning Posts by Tags"
taxonomy: machine-learning
author_profile: true
header:
   image: "/images/sky.jpg"

---

{% include group-by-array collection=site.posts | where:"categories","machine-learning" field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
	{% if post.categories contains "machine-learning" %}
		{% include archive-single.html %}
	{% endif %}
  {% endfor %}
{% endfor %}


