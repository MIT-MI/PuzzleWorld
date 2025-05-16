# Instructions for Submitting a Puzzle
To submit a puzzle, fork this repository and create a new branch. Then, create a new folder `{puzzle_name}` in the `data/puzzles` folder, and place the following files in it:

- `metadata.json`: A JSON file containing the metadata of the puzzle
- `content.png`: The image of the puzzle content
- `figure_{N}.png`: (Optional) Figures illustrating the reasoning steps

For an example puzzle, see the `data/puzzles/example` folder. After you are done, create a pull request to merge your branch into the main repository.

Note, please replace any spaces in the puzzle name with `_` when creating the new folder!

## Metadata
The `metadata.json` file should contain a JSON object with the following fields:

| Field Name   | Type       | Description                                      |
|--------------|------------|--------------------------------------------------|
| title        | string     | The title of the puzzle                          |
| flavor text  | string     | The flavor text of the puzzle, possibly empty    |
| difficulty   | string     | The difficulty level of the puzzle (easy, medium, hard) |
| solution     | string     | The solution to the puzzle                       |
| reasoning    | Step\[ \]  | An ordered list of reasoning [steps](#reasoning-step) towards the solution   |
| modality     | string\[ \] | A list of input [modalities](#a-list-of-input-modalities) the puzzle contains |
| skills       | string\[ \] | A list of [skills](#a-list-of-reasoning-skills) required to solve the puzzle    |
| source       | url        | Thel link to the puzzle                          |

### Reasoning Step
The `reasoning` field should contain a list of `Step` objects, which are represented as dictionaries with the following fields:
| Field Name   | Type      | Description                                            |
|--------------|-----------|--------------------------------------------------------|
| explanation  | string    | The textual explanation of the step                    |
| figure       | file path | (Optional) File path to a figure illustrating the step |

Each of the explanation should begin with one of the following atomic actions:
- Pattern discovery: discover patterns / insights from current information
  - E.g. discovering that current laser patterns are semaphores
- Sketching: sketching on or interacting with visual elements
  - E.g. traversing through a maze
  - E.g. connecting the dots
- Manipulation: manipulating or arranging a sequence of elements
  - E.g. sorting alphabets in order
  - E.g. applying cryptic encoding / decoding
- Combining / Chaining: combining or chaining multiple pieces of observations
  - E.g. matching patterns in images with text segments 
- Extraction: extracting information from one pattern or observation
  - E.g. extracting letters from semaphore patterns
 
(Note: the exact wording of action is not important as long as it resembles one of the above categories)
 
Each explanation step should consist of one action and the intermediate outcome of the action e.g. Identify the pattern that (...), which is (...)

### A List of Input Modalities
| Keyword        | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `text`         | Textual information                                                | 
| `visual`       | Unstructued visual information e.g. images, icons, fonts, etc.     |
| `structured`   | Structured visual information e.g. tables, graphs, crosswords, etc.|


### A List of Reasoning Skills
| Keyword        | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| `logic`        | Logic reasoning e.g. rule deduction or inferring conclusion given partial information |
| `wordplay`     | Manipulating words based on linguistic properties e.g. anagrams, homophones, etc.     |
| `spatial`      | Spatial or visual understanding, manipulation and navigation e.g. mazes, connecting dots, etc. |
| `cryptic`      | Encoding and decoding information e.g. ciphers, indexing, etc.                        | 
| `knowledge`    | Leverarging domain-specific knowledge e.g. history, science, etc.                     |
| `commonsense`  | Applying common sense reasoning e.g. physical laws, social norms, etc.                |
