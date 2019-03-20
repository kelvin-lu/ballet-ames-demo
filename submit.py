#!/usr/bin/env python

from textwrap import dedent

import ballet
import ballet.project
import ballet.update
import black
import click
import github
from ballet.compat import pathlib
from ballet.util.log import stacklog

import ames


def _make_branch_name(user, feature):
    return 'submit-feature-{user:02d}-{feature:02d}'.format(
        user=user, feature=feature)


def _check_environment(repo):
    assert repo.head.ref.name == ballet.update.DEFAULT_BRANCH
    assert not repo.is_dirty()


@stacklog(print, 'Creating and switching to new branch')
def create_and_switch_to_new_branch(repo, user, feature):
    name = _make_branch_name(user, feature)
    repo.create_head(name)
    repo.branches[name].checkout()


@stacklog(print, 'Creating subdirectories of contrib')
def create_dirs_if_needed(dst):
    dst = pathlib.path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)


@stacklog(print, 'Reformatting and writing feature')
def read_format_write(src, dst):
    with src.open('r') as f:
        code = f.read()
    code = black.format_file_contents(code)

    with dst.open('w') as f:
        f.write(code)


@stacklog(print, 'Adding init modules')
def add_init_if_needed(dst):
    for d in (dst.parent, dst.parent.parent):
        p = d.joinpath('__init__.py')
        p.touch(exist_ok=True)


@stacklog(print, 'Committing changes')
def commit_changes(repo):
    repo.git.add('.')
    repo.git.commit(m='Add new feature')


@stacklog(print, 'Pushing changes')
def push_changes(repo, user, feature):
    branch_name = _make_branch_name(user, feature)
    origin = repo.remote('origin')
    origin.push('{branch}:{branch}'.format(branch=branch_name))


def _make_github_client(token):
    if token is None:
        # read from ~/.github/token.txt
        p = pathlib.Path.home().joinpath('.github', 'token.txt')
        with p.open('r') as f:
            token = f.read().strip()

    g = github.Github(token)
    return g


@stacklog(print, 'Creating pull request')
def create_pull_request(gh, project, user, feature):
    branch_name = _make_branch_name(user, feature)
    owner = project.get('project', 'owner')
    name = project.get('project', 'name')
    ghrepo = gh.get_repo('{owner}/{name}'.format(owner=owner, name=name))

    # prepare PR metadata
    title = 'Propose new feature'
    body = dedent('''\
    Propose new feature: feature_{feature:02d}.py
    Submitted by user: user_{user:02d}.py
    
    Pull request automatically created by {script_name}
    ''').format(user=user, feature=feature, script_name='submit.py')
    base = ballet.update.DEFAULT_BRANCH
    head = branch_name
    maintainer_can_modify = True

    # create the pull
    pr = ghrepo.create_pull(
        title=title, body=body, base=base, head=head,
        maintainer_can_modify=maintainer_can_modify)

    return pr


@click.command()
@click.option('--user',
              required=True,
              type=click.INT,
              help='the "user" that created the feature (id between 1-9)')
@click.option('--feature',
              required=True,
              type=click.INT,
              help='the "name" of the feature (id between 1-n)')
@click.option('--from', 'from_',
              required=True,
              type=click.Path(exists=True,
                              file_okay=False,
                              readable=True,
                              resolve_path=True),
              help='root directory of existing features')
@click.option('--github-token',
              type=click.STRING,
              default=None,
              help='github access token to authorize pull request')
def submit(user, feature, from_, github_token):
    """Submit feature within path to project"""
    project = ballet.project.Project(ames)
    repo = project.repo
    to = project.get('contrib', 'module_path')

    src = pathlib.Path(from_,
                       'user_{user:02d}'.format(user=user),
                       'feature_{feature:02d}.py'.format(feature=feature))
    dst = pathlib.Path(to,
                       'user_{user:02d}'.format(user=user),
                       'feature_{feature:02d}.py'.format(feature=feature))

    _check_environment(repo)
    create_and_switch_to_new_branch(repo, user, feature)
    create_dirs_if_needed(dst)
    read_format_write(src, dst)
    add_init_if_needed(dst)
    commit_changes(repo)
    push_changes(repo, user, feature)
    gh = _make_github_client(github_token)

    pr = create_pull_request(gh, project, user, feature)
    print('Created pull request: {pr.url}'.format(pr=pr))

    print('Submission successful.')


if __name__ == '__main__':
    submit()
