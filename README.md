# SpaceTriangulation

Triangulate Space!!

![Example](usage.png)

## Example

```console 
> $ python main.py example_1.npz
> $ python main.py unit_cell.npz
```

## Controls

- `T`: Toggle tetrahedron mode. Select 3 point + selected point to create a tetrahedron
- `V`: Toggle visualization from: all, triangulation, dual
- `A`: Add a point in the center
- Left, Right arrows: move in `x` axis
- Up, Down arrows: move in `y` axis
- Shift+Up, Shift+Down arrows: move in `z` axis
- `I`: Show ids
- `G`: Save to `triangulation.npz` or path specified from cmd line
- `L`: Load from `triangulation.npz` or path specified from cmd line

[![Watch the video](usage.mp4)](usage.mp4)
