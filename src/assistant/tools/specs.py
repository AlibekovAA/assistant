from gigachat.models import Function, FunctionParameters
from gigachat.models.chat import FewShotExample, FunctionParametersProperty


def string_param(description: str, *, enum: list[str] | None = None) -> FunctionParametersProperty:
    return FunctionParametersProperty(type="string", description=description, enum=enum)


def make_function(
    *,
    name: str,
    description: str,
    properties: dict[str, FunctionParametersProperty],
    required: list[str] | None = None,
    examples: list[tuple[str, dict[str, object]]] | None = None,
    return_parameters: dict[str, object] | None = None,
) -> Function:
    few_shot = None
    if examples:
        few_shot = [FewShotExample(request=request, params=params) for request, params in examples]

    return Function(
        name=name,
        description=description,
        parameters=FunctionParameters(
            type="object",
            properties=properties,
            required=required,
        ),
        few_shot_examples=few_shot,
        return_parameters=return_parameters,
    )
