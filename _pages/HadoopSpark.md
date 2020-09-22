---
layout: category
permalink: /categories/hadoop-spark/
title: "Machine Learning Posts by Tags"
taxonomy: hadoop-spark
author_profile: true
header:
   image: "/images/HadoopSpark/Hadoop-Spark-Logo.png"

---

{% assign myposts = site.posts | where:"categories","hadoop-spark" %}
{% include group-by-array collection=myposts  field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
	{% if post.categories contains "hadoop-spark" %}
		{% include archive-single.html %}
	{% endif %}
  {% endfor %}
{% endfor %}


