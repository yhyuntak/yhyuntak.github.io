---
title : "Click-Through Rate Prediction"
layout : archive
permalink : categories/Click-Through_Rate_Prediction
author_profile : true
sidebar_main : true
---
<!-- 공백이 포함되어 있는 카테고리 이름의 경우 site.categories['a b c'] 이런식으로! -->

***

{% assign posts = site.categories['Click-Through Rate Prediction'] %} <!-- site.categories.example -->
{% for post in posts %} {% include archive-single_main.html type=page.entries_layout %} {% endfor %}
