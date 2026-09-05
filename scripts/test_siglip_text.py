from app.retrieval.visual_encoder import SigLIPEncoder


def main():
    encoder = SigLIPEncoder()

    print()
    print("Model class:")
    print(type(encoder.model))

    print()
    print("Model type:")
    print(encoder.model.config.model_type)

    print()
    print("Available feature methods:")

    methods = [
        name
        for name in dir(encoder.model)
        if "feature" in name.lower()
    ]

    for method in methods:
        print(f"- {method}")


if __name__ == "__main__":
    main()