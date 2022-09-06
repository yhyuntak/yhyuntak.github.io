---
title : "논문 리뷰"
layout : archive
permalink : categories/paper_review
author_profile : true
sidebar_main : true
---
<!-- 공백이 포함되어 있는 카테고리 이름의 경우 site.categories['a b c'] 이런식으로! -->

***

{% assign posts = site.categories['논문 리뷰'] %} <!-- site.categories.example -->
{% for post in posts %} {% include archive-single_main.html type=page.entries_layout %} {% endfor %}
