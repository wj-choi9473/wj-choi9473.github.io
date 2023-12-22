## Github Blog
- [https://wj-choi9473.github.io](https://wj-choi9473.github.io)
이 블로그는 minimal-mistakes theme 으로 만듬

### Structure

```
minimal-mistakes
├── _data                      # data files for customizing the theme
|  ├── navigation.yml          # main navigation links about, categoires등 추가
|  └── ui-text.yml             # text used throughout the theme's UI
├── _includes
|  ├── analytics-providers     # snippets for analytics (Google and custom)
|  ├── comments-providers      # snippets for comments 
|  ├── footer                  # custom snippets to add to site footer
|  ├── head                    # custom snippets to add to site head
|  ├── feature_row             # feature row helper
|  ├── gallery                 # image gallery helper
|  ├── group-by-array          # group by array helper for archives
|  ├── nav_list                # navigation list helper
|  ├── toc                     # table of contents helper
|  └── ...
├── _layouts
|  ├── archive-taxonomy.html   # tag/category archive for Jekyll Archives plugin
|  ├── archive.html            # archive base
|  ├── categories.html         # archive listing posts grouped by category
|  ├── category.html           # archive listing posts grouped by specific category
|  ├── collection.html         # archive listing documents in a specific collection
|  ├── compress.html           # compresses HTML in pure Liquid
|  ├── default.html            # base for all other layouts
|  ├── home.html               # home page
|  ├── posts.html              # archive listing posts grouped by year
|  ├── search.html             # search page
|  ├── single.html             # single document (post/page/etc)
|  ├── tag.html                # archive listing posts grouped by specific tag
|  ├── tags.html               # archive listing posts grouped by tags
|  └── splash.html             # splash page
├── _sass                      # SCSS partials
├── assets
|  ├── css
|  |  └── main.scss            # main stylesheet, loads SCSS partials from _sass
|  ├── images                  # image assets for posts/pages/collections/etc.
|  ├── js
|  |  ├── plugins              # jQuery plugins
|  |  ├── vendor               # vendor scripts
|  |  ├── _main.js             # plugin settings and other scripts to load after jQuery
|  |  └── main.min.js          # optimized and concatenated script file loaded before </body>
├── _config.yml                # site configuration 기본 설정이 저장된 파일, 자신에게 맞게 커스텀
├── Gemfile                    # gem file dependencies
├── index.html                 # paginated home page showing recent posts
└── package.json               # NPM build scripts

------------------------------------
주요 부분 요약
├── README.md
├── _config.yml : 기본 설정이 저장된 파일, 환경변수 설정파일
├── _includes : 기본 홈페이지 포맷(footer,head 등 변경)
├── _posts : 글 저장 폴더
├── _data : navigation.yml에서 네비게이션(about, categories 등등) 설정 
├── _pages/about.md : about에서 나타날 내용
├── assets : css, js, img 등 저장하는 폴더
```

- ```_config.yml```, ```_data/navigation.yml```, ```_pages```폴더 생성 및 필요한 .md파일 생성,  ```_pages/about.md``` 내용 수정
- ```assets/images/bio-photo``` 원하는 이미지로 설정


### 글 작성
-  ```_posts```에 해당하는 카테고리 디렉토리안에 ```categories:``` 설정한 후 글을 작성 (수식을 표현하려면 ```use_math: true```)
    - 새로운 카테고리를 만들어 글을 작성하고 싶다면 ```_posts```에 새로 디렉토리 생성 후 그 안에 글 작성 및 ```categories:``` 설정
- 글 파일이름은 ```YEAR-MONTH-DAY-titme.md | 2018-01-03-title1.md``` 이런 방식처럼 작성! 날짜를 빼고 쓰면 반영되지 않음. 또한, 제목은 영어로 작성
- 이미지를 추가할시 ```/assets/img/blog/''' 안에 폴더를 만든뒤 경로 설정



### Admin 세팅
admin page로 게시글을 쉽게 작성할 수 있음  
- Gemfile 파일에  gem ```'jekyll-admin', group: :jekyll_plugins``` 추가후 bundle install
- http://localhost:4000/admin 에 접속해서 게시물 쉽게 작성가능

### Latex tip
지킬에서 일부 latex문법이 지원되지 않으므로 식이 깨질때가 많음.
보통 깨지던 이유 보자면
1. ```|``` vertical var 를 그대로 사용하면 그렇다. ```\mid``` 나 원하는 vertical bar형태로 바꿀것
2. 식의 순서를 표현하기 위해 ```\tag{1}``` 를 사용하는데 이러면 또 깨진다... 그냥 ```.......(1)```  요런식으로 표현할것.
3. exponential 표현할때 e^x 가 깨지니 \exp(x) 사용
