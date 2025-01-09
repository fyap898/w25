# Tutorial 01: Data exploration and wrangling
Before building a machine learning model, it is important to understand and wrangle your data into an appropriate numeric format. In this tutorial, we'll look at how I like to set up my projects, some tips for exploratory visualizations, and (time allowing) how the marketing data from the AI course was wrangled.

## Part 1: Project configuration
Tutorials are not marked in this course, so it's up to you to keep track of them separately. I recommend copying the this directory to a new location rather than forking the entire `w25` repo and working out of that, otherwise you'll have a bunch of merge conflicts and extra stuff if you want to submit a PR to the main repo.

Tools:
- [Jupyter Notebooks](https://jupyter.org/)
- [Virtual environments](https://docs.python.org/3/library/venv.html)
- [Requirements files](https://pip.pypa.io/en/stable/user_guide/#requirements-files)
- [Git](https://git-scm.com/)
- [GitHub](https://github.com)

This semester, I'm going to try out using code reviews on GitHub as a way to both introduce you to an industry standard practice as well as highlight your contributions in assignments. To test this out, I've created a [practice repo](https://github.com/mru-comp4630-w25/pr_practice) where you can:
1. Fork a copy to your own GitHub account
2. Make edits to the code
3. Submit a pull request (PR)
4. Review each other's changes

## Part 2: Exploratory visualizations
Follow along with the [notebook](visualization_tips.ipynb) and answer the various TODOs.

## Part 3: Revisit the Bank Marketing data
1. Create a new .ipynb file to explore this new dataset
2. Read the [raw data](https://archive.ics.uci.edu/dataset/222/bank+marketing) into a pandas DataFrame. You can either download the zip file, or install the `ucimlrepo` package and fetch the data directly.
3. Read the [pre-processed version](marketing_cleaned.csv) into a different pandas DataFrame.
4. Try to answer the following questions:
    1. How were the categorical features handled?
    2. Were any of the numerical categories manipulated?
    3. What additional transformations might be useful for this dataset?