# Java Developer Copilot Skills & Extensions

## 1. Native Java Power Moves (Copilot Core)
*   **Boilerplate & POJO Generation**: Auto-complete constructors, getters, setters, `equals()`, and `hashCode()` instantly.
*   **Spring Boot Autowiring**: Automatically suggests `@Autowired` dependencies and standard Spring MVC/REST controller boilerplate.
*   **Regex to Java**: Explain complex Regex patterns into `java.util.regex` implementations via inline chat (`Cmd+I` / `Ctrl+I`).
*   **SQL for JPA/Hibernate**: Type native SQL logic or JPQL in natural language; Copilot formats it into proper `@Query` annotations.

## 2. Testing & Quality (Essential)
*   **JUnit / Mockito Generator**: Highlight any Java method, use `/tests`, and Copilot will generate complete JUnit 5 test classes with mocked dependencies (Mockito).
*   **SonarLint (Extension)**: Pairs with Copilot to ensure generated Java code passes standard static analysis and security gates before commit.
*   **Sentry for Copilot (Extension)**: Automatically links stack traces from production logs to your local Java files and suggests fixes.

## 3. DevOps & Containerization
*   **Docker for GitHub Copilot**: Generate optimized multi-stage `Dockerfile` and `docker-compose.yml` for Maven/Gradle Spring Boot apps. Ask `@workspace "how to containerize this Java app?"`

## 4. Documentation & Onboarding
*   **JavaDoc Auto-Gen**: Highlight a class or complex method, use `/doc` to generate standard JavaDoc tags (`@param`, `@return`, `@throws`).
*   **PerplexityAI (Extension)**: Search the latest Spring Framework or Java 21 documentation directly within your IDE chat, bypassing outdated training data.

## 5. Build Tools & Dependencies
*   **Maven/Gradle Assistant**: Describe the library you need (e.g., "Add Jackson JSON parser"), and Copilot injects the exact XML/Groovy dependency block with the latest stable version into your `pom.xml` or `build.gradle`.
