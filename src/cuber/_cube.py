"""
NOTE: This is copied from David W. Hogg's https://github.com/davidwhogg/MagicCube - but only keeping portions for plotting cube in notebook environment


license
-------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301 USA.

usage
-----
- initialize a solved cube with `c = Cube(N)` where `N` is the side length.
- todo: complete this...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import re
from ._cube_painter import draw_cube, render_horiz
from copy import deepcopy

class Cube(object):
    """
    Cube
    ----
    Initialize with arguments:
    - `N`, the side length (the cube is `N`x`N`x`N`)
    - optional `whiteplastic=True` if you like white cubes
    """
    facedict = {"U":0, "D":1, "F":2, "B":3, "R":4, "L":5}
    dictface = dict([(v, k) for k, v in facedict.items()])
    normals = [np.array([0., 1., 0.]), np.array([0., -1., 0.]),
               np.array([0., 0., 1.]), np.array([0., 0., -1.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.])]
    # this xdirs has to be synchronized with the self.apply_move() function
    xdirs = [np.array([1., 0., 0.]), np.array([1., 0., 0.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
               np.array([0., 0., -1.]), np.array([0, 0., 1.])]
    colordict = {"w":0, "y":1, "b":2, "g":3, "o":4, "r":5}
    pltpos = [(0., 1.05), (0., -1.05), (0., 0.), (2.10, 0.), (1.05, 0.), (-1.05, 0.)]
    labelcolor = "#7f00ff"

    def __init__(self, N, whiteplastic=False, echo=True):
        """
        (see above)
        """
        self.N = N
        self.echo = echo
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.stickercolors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        self.stickerthickness = 0.001 # sticker thickness in units of total cube size
        self.stickerwidth = 0.9 # sticker size relative to cubie size (must be < 1)
        self.movelist = []
        if whiteplastic:
            self.plasticcolor = "#dfdfdf"
        else:
            self.plasticcolor = "#1f1f1f"
        self.fontsize = 12. * (self.N / 5.)
        self.draw_front = True
        # self.draw_top = True
        self.draw_top = False
        self.top_solving = 'pll'

    def copy(self):
        return deepcopy(self)

    def reset(self):
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        return self

    def turn(self, f): # , d=1):
        """
        Turn whole cube (without making a layer move) around face `f`
        `d` 90-degree turns in the clockwise direction.  Use `d=3` or
        `d=-1` for counter-clockwise.
        """
        assert re.match(r"[RUFLDBrufldb](?:[23]'?|'[23]?)?", f)
        self.movelist.append(('turn', f))
        if self.echo:
            print(f"Turn: {f}")
        d = 1
        reverse = False
        face = f[0]
        for modifier in f[1:]:
            if modifier== "'":
                reverse = True
            else:
                d = int(modifier)  
        if reverse:
            d = 4 - d
        for l in range(self.N):
            self.apply_move(face, l, d, store_in_movelist=False)
        return self

    def move(self, move_string, echo = True, debug=False):
        moves = re.findall(r"[xyzMESRUFLDBrufldb](?:[23]'?|'[23]?)?", move_string)
        self.movelist.append(('move', ' '.join(moves)))
        if self.echo:
            print(' '.join(moves))
        for move in moves:
            if debug:
                print(move)
            if move[0] in 'xyz':
                d = 1
                reverse = False
                face = {'x': 'R', 'y': 'U', 'z': 'F'}[move[0]]
                for modifier in move[1:]:
                    if modifier== "'":
                        reverse = True
                    else:
                        d = int(modifier)  
                if reverse:
                    d = 4 - d
                for layer in range(self.N):
                    if debug:
                        print(f"Calling self.apply_move({face}, {layer}, {d})")
                    self.apply_move(face, layer, d, store_in_movelist=False)
                continue
            elif move[0] not in 'MES':
                face = move[0]
                layers = [0]
            else:
                if move[0] == 'M':
                    face = 'L'
                    layers = [1]
                elif move[0] == 'E':
                    face = 'D'
                    layers = [1]
                elif move[0] == 'S':
                    face = 'F'
                    layers = [1]
            d = 1
            reverse = False
            if face.islower():
                layers.append(1)
                face = face.upper()
            for modifier in move[1:]:
                if modifier== "'":
                    reverse = True
                else:
                    d = int(modifier)
            if reverse:
                d = 4 - d
            for layer in layers:
                if debug:
                    print(f"Calling self.apply_move({face}, {layer}, {d})")
                self.apply_move(face, layer, d, debug=debug, store_in_movelist=False)
        return self
                
    def undo(self, debug=False):
        kind, move = self.movelist.pop()
        if debug:
            print(f"kind = {kind}, move = {move}")
        if kind=='move':
            self.reverse(move)
        else:
            raise NotImplementedError("To be done...")
        return self
                        
    def get_reverse_moves(self, move_string):
        moves_result = []
        moves = re.findall(r"[MESRUFLDBrufldb](?:[23]'?|'[23]?)?", move_string)
        for move in reversed(moves):
            face = move[0]
            d = 1
            reverse = True # opposite
            for modifier in move[1:]:
                if modifier== "'":
                    reverse = False # opposite
                else:
                    d = int(modifier)
            thismove = [face]
            if reverse:
                thismove.append("'")
            if d>1:
                thismove.append(f"{d}")
            thismove = ''.join(thismove)
            moves_result.append(thismove)
        return ' '.join(moves_result)
    
    def reverse(self, move_string):
        reversed_moves = self.get_reverse_moves(move_string)
        if self.echo:
            print("Here")
            print(f'reversed({move_string}) ==> ', end='')
        self.move(reversed_moves)
        return self
    
    def apply_move(self, f, l=0, d=1, debug=False, store_in_movelist=True):
        """
        Make a layer move of layer `l` parallel to face `f` through
        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `l` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        """
        if store_in_movelist:
            self.movelist.append(('apply_move', (f, l, d)))
        i = self.facedict[f]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["F"], range(self.N), l2),
                              (self.facedict["R"], range(self.N), l2),
                              (self.facedict["B"], range(self.N), l2),
                              (self.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.apply_move("U", l2, -d, store_in_movelist=store_in_movelist)
        if f == "F":
            f2 = "B"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], range(self.N), l),
                              (self.facedict["L"], l2, range(self.N)),
                              (self.facedict["D"], range(self.N)[::-1], l2),
                              (self.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.apply_move("F", l2, -d, store_in_movelist=store_in_movelist)
        if f == "R":
            f2 = "L"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], l2, range(self.N)),
                              (self.facedict["F"], l2, range(self.N)),
                              (self.facedict["D"], l2, range(self.N)),
                              (self.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.apply_move("R", l2, -d, store_in_movelist=store_in_movelist)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        if debug:
            print("moved", f, l, len(ds))
        return self

    def _rotate(self, args):
        """
        Internal function for the `move()` function.
        """
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def move_random(self, number):
        """
        Make `number` randomly chosen moves to scramble the cube.
        """
        for t in range(number):
            f = self.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            d = 1 + np.random.randint(3)
            self.apply_move(f, l, d)
        return self

    def _render_points(self, points, viewpoint):
        """
        Internal function for the `render()` function.  Clunky
        projection from 3-d to 2-d, but also return a zorder variable.
        """
        v2 = np.dot(viewpoint, viewpoint)
        zdir = viewpoint / np.sqrt(v2)
        xdir = np.cross(np.array([0., 1., 0.]), zdir)
        xdir /= np.sqrt(np.dot(xdir, xdir))
        ydir = np.cross(zdir, xdir)
        result = []
        for p in points:
            dpoint = p - viewpoint
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1. * viewpoint)
            result += [np.array([np.dot(xdir, dproj),
                                 np.dot(ydir, dproj),
                                 np.dot(zdir, dpoint / np.sqrt(v2))])]
        return result
    
    def _render_view(self, ax, view_angle, shift_x, shift_y):        
        if view_angle == 'flat':
            """
            Make an unwrapped, flat view of the cube for the `render()`
            function.  This is a map, not a view really.  It does not
            properly render the plastic and stickers.
            """
            for f, i in self.facedict.items():
                x0, y0 = self.pltpos[i]
                x0 -= 1.2
                y0 += 1.5
                x0 += shift_x
                y0 += shift_y
                cs = 1. / self.N
                for j in range(self.N):
                    for k in range(self.N):
                        ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs),
                                                cs, cs, ec=self.plasticcolor,
                                                fc=self.stickercolors[self.stickers[i, j, k]]))
                ax.text(x0 + 0.5, y0 + 0.5, f, color=self.labelcolor,
                        ha="center", va="center", rotation=20, fontsize=self.fontsize)
            return
        csz = 2. / self.N
        x2 = 8.
        x1 = 0.5 * x2             
        viewpoint_shifts = {
            "bottom-left": (np.array([-x1, -x1, x2]), np.array([-1.5, 3.])), # np.array([-1.5, 3.])),
            "front-right": (np.array([x1, x1, x2]), np.array([-1.5, 3.])), # np.array([0.5, 3.])),
            "right-back": (np.array([x2, x1, -x1]), np.array([-1.5, 3.])), # np.array([2.5, 3.]))
        }
        if view_angle not in viewpoint_shifts:
            raise ValueError("Error not in {['flat'] + list(viewpoint_shifts.keys())}")
        viewpoint, shift = viewpoint_shifts[view_angle]
        shift[0] += shift_x
        shift[1] += shift_y
        for f, i in self.facedict.items():
            zdir = self.normals[i]
            if np.dot(zdir, viewpoint) < 0:
                continue
            xdir = self.xdirs[i]
            ydir = np.cross(zdir, xdir) # insanity: left-handed!
            psc = 1. - 2. * self.stickerthickness
            corners = [psc * zdir - psc * xdir - psc * ydir,
                       psc * zdir + psc * xdir - psc * ydir,
                       psc * zdir + psc * xdir + psc * ydir,
                       psc * zdir - psc * xdir + psc * ydir]
            projects = self._render_points(corners, viewpoint)
            xys = [p[0:2] + shift for p in projects]
            zorder = np.mean([p[2] for p in projects])
            ax.add_artist(Polygon(xys, ec="none", fc=self.plasticcolor))
            for j in range(self.N):
                for k in range(self.N):
                    corners = self._stickerpolygon(xdir, ydir, zdir, csz, j, k)
                    projects = self._render_points(corners, viewpoint)
                    xys = [p[0:2] + shift for p in projects]
                    ax.add_artist(Polygon(xys, ec="none", fc=self.stickercolors[self.stickers[i, j, k]]))
            x0, y0, zorder = self._render_points([1.5 * self.normals[i], ], viewpoint)[0]
            ax.text(x0 + shift[0], y0 + shift[1], f, color=self.labelcolor,
                    ha="center", va="center", rotation=20, fontsize=self.fontsize / (-zorder))        


    def _stickerpolygon(self, xdir, ydir, zdir, csz, j, k):
        small = 0.5 * (1. - self.stickerwidth)
        large = 1. - small
        return [zdir - xdir + (j + small) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + large - small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small) * csz * xdir - ydir + (k + large - small) * csz * ydir]


    def render_fig(self, *views, scale=1.5):
        """
        Visualize the cube in a standard layout, including a flat,
        unwrapped view and three perspective views.
        
        view should be in 'bottom-left', 'front-right', 'right-back', or 'flat'
        """
        if len(views)==0:
            views = ['front-right']
        # ylim = (-1.2, 4.)
        if isinstance(views, str):
            views = [views]
        xlim = [-2.4, -1.9]
        ylim = [0.2, 4]
        for view in views:
            if view == 'flat':
                xlim[1] += 4.5
            else:
                xlim[1] += 2
        if 'flat' not in views:
            ylim[0] += 1.9
        fig = plt.figure(figsize=((xlim[1] - xlim[0]) * self.N / 5.,
                                  (ylim[1] - ylim[0]) * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False,
                              xticks=[], yticks=[])
        shift_x = 0
        shift_y = 0
        for view in views:
            self._render_view(ax, view, shift_x=shift_x, shift_y=shift_y)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if view == 'flat':
                shift_x += 4.5
            else:
                shift_x += 2
        width, height = fig.get_size_inches()
        fig.set_size_inches(width*scale, height*scale)
        return fig

    def _render(self, perspectives, top_solving):
        if isinstance(perspectives, str):
            perspectives = [perspectives]
        images = []
        for perspective in perspectives:
            if perspective == 'top-view':
                # sticker_cmap = 'oll' if top_solving=='oll' else None
                sticker_cmap = top_solving
                images.append(draw_cube(self.stickers, perspective, sticker_cmap=sticker_cmap))
            else:
                images.append(draw_cube(self.stickers, perspective))
        return render_horiz(*images)
    
    def render(self):
        perspectives = []
        if self.draw_front:
            perspectives.append('front-right')
        if self.draw_top:
            perspectives.append('top-view')
        pil_image = self._render(perspectives=perspectives, top_solving=self.top_solving)
        return pil_image

    def render_old(self, *views, scale=1.5):
        fig = self.render_fig(*views, scale=scale)
        plt.show(fig)
        
    def _ipython_display_(self):
        display(self.render())
        
