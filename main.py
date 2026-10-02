import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.collections as mp_coll
import matplotlib.text as mp_text
import matplotlib.axes as mp_axes
import matplotlib.figure as mp_figure
from mpl_toolkits.mplot3d import art3d, Axes3D
from typing import List, Literal, Optional as Op, Tuple
from dataclasses import dataclass
import sys

FArray = npt.NDArray[np.float32]
IArray = npt.NDArray[np.int32]
Ax = mp_axes.Axes
Fig = mp_figure.Figure
Text = mp_text.Text

def calculate_normals(points: FArray, triangles: IArray) -> FArray:
    normals = np.zeros((triangles.shape[0], points.shape[1]), np.float32)
    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]
        vs = np.zeros((triangles.shape[1], points.shape[1]))
        for j in range(vs.shape[0]):
            p0 = points[triangle[j]]
            pf = points[triangle[(j+1)%len(triangle)]]
            vs[j, :] = pf - p0
        assert np.all(np.sum(vs, axis=0) < 1e-10)
        normals[i] = n = 1/2 * np.cross(vs[0], vs[1])
        assert np.all([abs(np.dot(n, vs[i])) < 1e-10 for i in range(vs.shape[0])])
    return normals

def plot_segments(points: FArray, segments: IArray, ax: Axes3D) -> None:
    for i in range(segments.shape[0]):
        s = segments[i]
        ax.plot3D(
            points[s, 0], points[s, 1], points[s, 2], 
            linewidth=1,
            color="black",
            alpha=0.5,
        )

def plot_points(
    points: FArray, ax: Axes3D,
    *,
    selected_idx: Op[int],
    secondary_selection: List[int],
    draw_ids: bool,
    nodes: IArray,
    tetrahedra: IArray,
) -> mp_coll.PathCollection:
    boundary = []
    # for i in range(points.shape[0]):
    #     for j in range(tetrahedra.shape[0]):
    #         if i in tetrahedra[j] and j in nodes:
    #             boundary.append(i)

    colors = [
        "purple" 
        if i in secondary_selection else
        "orange"
        if i == selected_idx else 
        "red"
        # if i in boundary else
        # "grey"
        for i in range(points.shape[0])
    ]


    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],  # pyright: ignore[reportArgumentType]
        marker=".",
        s=100,
        facecolor=colors,
        edgecolor="black",
        picker=5,
    )
    if draw_ids:
        for i in range(points.shape[0]):
            x, y, z = points[i]
            if i == selected_idx:
                ax.text(x, y, z, f"p[{i}] = {x:.02f}, {y:.02f}, {z:.02f}")
            else:
                ax.text(x, y, z, f"p[{i}]")
    return scatter


def plot_triangles(
    points: FArray, triangles: IArray,
    ax: Axes3D,
    *,
    draw_segment_vectors: bool,
    draw_normals: bool,
    draw_ids: bool,
    nodes: IArray,
    tetrahedra: IArray
) -> List[Text]:
    texts = []
    boundary = []
    for i in range(triangles.shape[0]):
        for j in range(tetrahedra.shape[0]):
            if all(p in tetrahedra[j, :] for p in triangles[i, :]) and j in nodes:
                boundary.append(i)
                break
        
    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]
        if draw_segment_vectors:
            for j in range(triangle.shape[0]):
                p0 = points[triangle[j]]
                pf = points[triangle[(j+1)%len(triangle)]]
                v = pf - p0
                ax.quiver3D(
                    p0[0], p0[1], p0[2],
                    v[0], v[1], v[2],
                )
        center = np.mean(points[triangle], axis=0)
        if draw_normals:
            n = calculate_normals(points, triangles[i])[0]
            ax.quiver3D(
                center[0], center[1], center[2],
                n[0], n[1], n[2],
                color="black",
                linewidth=0.5,
                arrow_length_ratio=0.1,
            )
        if i in boundary:
            ax.add_collection(
                art3d.Poly3DCollection(
                    [points[triangle]],
                    facecolor="yellow",
                    alpha=0.08,
                )
            )
        if draw_ids:
            ax.text(center[0], center[1], center[2], f"t[{i}]")
    return texts

def calculate_vertices(
    points: FArray,
    tetrahedra: IArray,
) -> FArray:
    vertices = []
    for i in range(tetrahedra.shape[0]):
        v = np.mean(points[tetrahedra[i], :], axis=0)
        vertices.append(v)
    return np.array(vertices)

def calculate_edges(tetrahedra: IArray) -> IArray:
    edges = []
    for i in range(tetrahedra.shape[0]):
        tet_i = tetrahedra[i, :]
        for j in range(tetrahedra.shape[0]):
            if j >= i:
                continue
            tet_j = tetrahedra[j, :]
            num_common = sum(p in tet_i for p in tet_j)
            if num_common == 3:
                edges.append([j, i])
    return np.array(edges)

def calculate_boundary_graph(
    edges: IArray,
    tetrahedra: IArray,
) -> Tuple[IArray, IArray]:
    nodes = []
    links = []
    for i in range(tetrahedra.shape[0]):
        tet = tetrahedra[i, :]
        t1 = sorted(tuple(tet[[0, 1, 2]]))
        t2 = sorted(tuple(tet[[0, 2, 3]]))
        t3 = sorted(tuple(tet[[0, 3, 1]]))
        t4 = sorted(tuple(tet[[1, 3, 2]]))
        boundary = 4
        for t in (t1, t2, t3, t4):
            for j in range(tetrahedra.shape[0]):
                if i == j:
                    continue
                tet_j = tetrahedra[j, :]
                if all(p in tet_j for p in t):
                    boundary -= 1
                    break
        assert boundary >= 0
        if boundary > 0:
            nodes.append(i)
    for i in range(edges.shape[0]):
        ed = edges[i]
        if ed[0] in nodes and ed[1] in nodes:
            links.append(i)
    return np.array(nodes), np.array(links)


def plot_vertices(
    vertices: FArray, ax: Axes3D,
    *,
    nodes: IArray,
    draw_ids: bool,
) -> None:
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        if i in nodes:
            color = (.3, .3, 1)
        else:
            color = "green"
        ax.scatter(
            x, y, z,
            marker=".",
            s=100,
            facecolor=color,
            edgecolor="black",
            picker=5,
        )
        if draw_ids:
            ax.text(x, y, z, f"v[{i}]")

def plot_edges(
    vertices: FArray,
    edges: IArray,
    ax: Axes3D,
    *,
    links: IArray,
) -> None:
    for i in range(edges.shape[0]):
        s = edges[i]
        if i in links:
            color = "green"
        else:
            color = "blue"
        ax.plot3D(
            vertices[s, 0], vertices[s, 1], vertices[s, 2], 
            linewidth=1,
            color=color,
            alpha=0.5,
        )


@dataclass
class State:
    points: FArray
    tetrahedra: IArray
    selected_point: Op[int]
    secondary_selection: List[int]
    selection_mode: Literal["none", "tetrahedra"]
    scatter_collection: Op[mp_coll.PathCollection]
    draw_ids: bool
    draw_normals: bool
    draw_segment_vectors: bool
    view_mode: Literal["all", "triangulation", "dual"]

    def calculate_triangles(self) -> IArray:
        triangles = []
        for i in range(self.tetrahedra.shape[0]):
            tet = self.tetrahedra[i, :]
            t1 = sorted(tuple(tet[[0, 1, 2]]))
            t2 = sorted(tuple(tet[[0, 2, 3]]))
            t3 = sorted(tuple(tet[[0, 3, 1]]))
            t4 = sorted(tuple(tet[[1, 3, 2]]))
            for t in (t1, t2, t3, t4):
                if t not in triangles:
                    triangles.append(t)
        return np.array([list(t) for t in triangles])

    def calculate_segments(self) -> IArray:
        segments = []
        for i in range(self.tetrahedra.shape[0]):
            tet = self.tetrahedra[i, :]
            segs = [
                sorted(tuple(tet[[i, j]]))
                for i in range(tet.shape[0])
                for j in range(tet.shape[0])
                if i < j
            ]
            for s in segs:
                if s not in segments:
                    segments.append(s)
        return np.array([list(seg) for seg in segments])


def main() -> None:
    # points0 = np.array([  # List of (x, y, z) coords
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [-1/2, np.sqrt(3)/2, 0],
    #     [-1/2, -np.sqrt(3)/2, 0],
    #
    #     [-1.5, 0, 1],
    # ])
    # tetrahedra0 = np.array([  # List of (p1, p2, p3, p4) indices in points
    #     [0, 1, 2, 3],
    #     [0, 2, 3, 4],
    # ])
    points0 = np.array([
        [-1, -1, -1],
        [ 0, -1, -1],
        [ 1, -1, -1],

        [-1,  0, -1],
        [ 0,  0, -1],
        [ 1,  0, -1],
        
        [-1,  1, -1],
        [ 0,  1, -1],
        [ 1,  1, -1],

        [-1, -1,  0],
        [ 0, -1,  0],
        [ 1, -1,  0],

        [-1,  0,  0],
        [ 0,  0,  0],
        [ 1,  0,  0],
        
        [-1,  1,  0],
        [ 0,  1,  0],
        [ 1,  1,  0],

        [-1, -1,  1],
        [ 0, -1,  1],
        [ 1, -1,  1],

        [-1,  0,  1],
        [ 0,  0,  1],
        [ 1,  0,  1],
        
        [-1,  1,  1],
        [ 0,  1,  1],
        [ 1,  1,  1],
    ])
    tetrahedra0 = np.array([
    ])

    state = State(
        points=points0,
        tetrahedra=tetrahedra0,
        selected_point=None,
        secondary_selection=[],
        selection_mode="none",
        scatter_collection=None,
        draw_ids=False,
        draw_normals=False,
        draw_segment_vectors=False,
        view_mode="all",
    )
    try:
        filename = sys.argv[1]
        print(f"Using filename: {filename}")
        d = np.load(filename)
        state.points = d["points"]
        state.tetrahedra = d["tetrahedra"]
    except IndexError:
        filename = "triangulation.npz"

    def update_plot():
        """Refreshes the scatter collection and updates colors based on selection."""
        # for s in data["scatters"]:
        #     s.remove()  # Clear old scatter object
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        # Highlight the selected point in red, others in blue
        vertices = calculate_vertices(state.points, state.tetrahedra)
        edges = calculate_edges(state.tetrahedra)
        nodes, links = calculate_boundary_graph(edges, state.tetrahedra)
                
        if state.view_mode in ("all", "dual"):
            plot_edges(vertices, edges, ax, links=links)
            plot_vertices(vertices, ax, nodes=nodes, draw_ids=state.draw_ids)

        if state.view_mode in ("all", "triangulation"):
            plot_triangles(
                state.points, state.calculate_triangles(), ax,
                draw_segment_vectors=state.draw_segment_vectors,
                draw_normals=state.draw_normals,
                draw_ids=state.draw_ids,
                nodes=nodes,
                tetrahedra=state.tetrahedra,
            )
            plot_segments(state.points, state.calculate_segments(), ax)
            state.scatter_collection = plot_points(
                state.points, ax,
                selected_idx=state.selected_point,
                secondary_selection=state.secondary_selection,
                draw_ids=state.draw_ids,
                nodes=nodes,
                tetrahedra=state.tetrahedra,
            )

        ax.text2D(0.05, 0.95, f"Selection Mode: {state.selection_mode}", transform=ax.transAxes)
        fig.canvas.draw_idle()

    def on_pick(event):
        """Triggered when clicking on an existing point."""
        if event.artist == state.scatter_collection:
            i = event.ind[0]
            if i == state.selected_point:
                state.selected_point = None
            elif state.selected_point is None or state.selection_mode == "none":
                state.selected_point = i
            elif state.selection_mode == "tetrahedra":
                if i in state.secondary_selection:
                    state.secondary_selection.remove(i)
                elif len(state.secondary_selection) < 3:
                    state.secondary_selection.append(i)
                else:
                    assert False, "Unreachable"

                if len(state.secondary_selection) == 3:
                    tet = [state.selected_point, *state.secondary_selection]
                    assert all(p is not None for p in tet), tet
                    tet2 = []
                    for j in range(state.tetrahedra.shape[0]):
                        t = list(state.tetrahedra[j, :])
                        if tet is not None and all(p in tet for p in t):
                            tet = None
                            continue
                        tet2.append(t)
                    if tet is not None:
                        tet2.append(tet)
                    state.tetrahedra = np.array([[p0, p1, p2, p3] for p0, p1, p2, p3 in tet2])
                    state.secondary_selection = []
            else:
                assert False, "Unreachable"
        update_plot()

    def on_key(event):
        """Moves the selected point using Arrow Keys."""
        if event.key == 'A':  # Add point
            p = np.array([[0, 0, 0]])
            state.points = np.concatenate([state.points, p])
            state.selected_point = state.points.shape[0] - 1
        elif event.key == 'T':
            state.selection_mode = "tetrahedra" if state.selection_mode != "tetrahedra" else "none"
            state.secondary_selection = []
        elif event.key == 'I':
            state.draw_ids = not state.draw_ids
        elif event.key == 'V':
            if state.view_mode == "all":
                state.view_mode = "dual"
            elif state.view_mode == "dual":
                state.view_mode = "triangulation"
            elif state.view_mode == "triangulation":
                state.view_mode = "all"
            else:
                assert False, "Unreachable"
        elif event.key == 'G':
            np.savez(
                filename,
                points=state.points,
                tetrahedra=state.tetrahedra,
            )
            print(f"Saved triangulation to file: {filename}")
        elif event.key == 'L':  # Toggle segment mode
            d = np.load(filename)
            state.points = d["points"]
            state.tetrahedra = d["tetrahedra"]
        
        if state.selected_point is not None:
            step = 0.1  # Movement speed increments
            dx = dy = dz = 0
            if event.key == 'left':
                dx -= step
            elif event.key == 'right':
                dx += step
            elif event.key == 'up':
                dy += step
            elif event.key == 'down':
                dy -= step
            elif event.key == 'shift+up':
                dz += step
            elif event.key == 'shift+down':
                dz -= step
            state.points[state.selected_point, :] += [dx, dy, dz]

        update_plot()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    update_plot()
    # Connect the interactive event managers
    fig.canvas.mpl_connect('pick_event', on_pick)
    # fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == "__main__":
    main()
